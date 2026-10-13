import os
import torch
import numpy as np
import random
import copy
########################################################################
seed_num = 42
def set_seed(seed=seed_num):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed_num)

tg = torch.Generator()
tg.manual_seed(seed_num)
########################################################################
from neural_networks import NeuralNetworkPotential, BaselineParameters
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

from module import (
    device,
    dtype,
    r_cut,
    Ha2mev,
    Ha_Bohr2meV_angstrom,
    model_type,
    dataset_key1,
    dataset_key2,
    ANG2BOHR
)
from tqdm import tqdm
from train_helper import atom_mse_to_graph, compute_batch_predictions, evaluate_validation_batch
########################################################################
# Loss weights w of Eqs. (4)-(5)
w_E     = 10.0      # energy per atom
w_F     = 1000.0    # forces
w_q     = 10.0      # MBIS charges, L_aux
w_p     = 0.0       # atomic dipoles are NOT supervised (latent)
w_mu    = 0.1       # molecular dipole
w_alpha = 0.0001    # molecular polarizability
num_epochs = 30000

# Separation windows (Å) for the baseline loss L(E, E_b): applied to dimers with d_lr_min <= d <= d_train_max
d_train_max = 7.1
d_lr_min    = 4.7
########################################################################

def keep_training_fields_only(data):
    keep = {"pos", "species", "forces", "E_tot", "charge_tot", "charges", "dipoles", "dipole",
            "polarizability", "d", "edge_index", "train_loss_mode"}
    for key in list(data.keys()):
        if key not in keep:
            del data[key]
    return data

def prepare(data):
    # positions and molecular-dipole labels are used in the frame stored in the dataset
    # (the label and the predicted dipole of a charged monomer share the same origin)
    keep_training_fields_only(data)
    data.edge_index = radius_graph(data.pos, r=r_cut, loop=False)
    return data

dataset_dimer_all = torch.load(dataset_key1, weights_only=False)
dataset_monomer_all = torch.load(dataset_key2, weights_only=False)

# training dimers: 3.5 <= d <= 7.1 Å at zero field
dataset_dimer = [copy.deepcopy(data) for data in dataset_dimer_all if float(data.d.item()) / ANG2BOHR <= d_train_max + 1e-6]
print(f"Using {len(dataset_dimer)} / {len(dataset_dimer_all)} DIMER TRAIN samples with d <= {d_train_max} Å.")
for data in tqdm(dataset_dimer):
    prepare(data)
    d_ang = float(data.d.item()) / ANG2BOHR
    # train_loss_mode 0: L(E, E_tot) only;  1: L(E, E_tot) + L(E, E_b)
    data.train_loss_mode = torch.tensor([1 if d_ang >= d_lr_min else 0])

# isolated monomers of the training templates
dataset_monomer = [copy.deepcopy(data) for data in dataset_monomer_all]
print(f"Using {len(dataset_monomer)} MONOMER TRAIN samples.")
for data in tqdm(dataset_monomer):
    prepare(data)
    data.train_loss_mode = torch.tensor([0])

dataset = dataset_dimer + dataset_monomer
print(f"Loaded TRAIN dataset with {len(dataset)} samples ({len(dataset_dimer)} dimer + {len(dataset_monomer)} monomer).")

# checkpoint-selection set: the training dimer configurations (SI Note 3)
dataset_validation = [prepare(copy.deepcopy(data)) for data in dataset_dimer_all if float(data.d.item()) / ANG2BOHR <= d_train_max + 1e-6]
print(f"Loaded VALIDATION dataset with {len(dataset_validation)} samples.")

batch_size = 15
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=tg)
validation_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
########################################################################
nnp             = NeuralNetworkPotential().to(device=device, dtype=dtype)
baseline_params = BaselineParameters().to(device=device, dtype=dtype)

num_nn_params = sum(p.numel() for p in nnp.parameters())
num_b_params  = sum(p.numel() for p in baseline_params.parameters())
print("[num params - MPNN]:", num_nn_params, " [num params - baseline]:", num_b_params)

optimizer = torch.optim.Adam(list(nnp.parameters()) + list(baseline_params.parameters()), lr=2e-3)

# ===================== Check point load =====================
checkpoint_path = "nnp_checkpoint.pt"
# state-dict keys renamed during the code clean-up (no change in the computation)
_renamed_keys = {"radial_basis.bpi": "radial_basis.kappa"}
_dropped_prefixes = ("node_embedding.muq_lin.",)   # unused parameters removed

def _migrate_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith(_dropped_prefixes):
            continue
        for old, new in _renamed_keys.items():
            k = k.replace(old, new)
        out[k] = v
    return out

start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    nnp.load_state_dict(_migrate_state_dict(checkpoint['nnp_state_dict']))
    baseline_params.load_state_dict(checkpoint['dp_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']
    validation_metric = checkpoint['validation_metric']
else:
    print("No checkpoint found. Starting fresh train.")
    history = [[], [], [], [], []]            # 0: loss, 1: loss_E, 2: loss_F, 3: |dE|, 4: |dF|
    validation_metric = [[], [], [], [], []]  # 0: Energy RMSE, 1: Force RMSE, 2: Mu RMSE, 3: Alpha RMSE, 4: R2
# ====================================================================
red = "\033[31m"
reset = "\033[0m"
print(f"dataset: {red}{dataset_key1} + {dataset_key2}{reset}")
print(f"model_type: {red}{model_type}{reset}")

for epoch in range(start_epoch, num_epochs):
    nnp.train()
    baseline_params.train()
    total_loss_sum, loss_E_sum, loss_F_sum, diff_E_sum, diff_F_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    num_batches = 0
    for batch in train_loader:
        batch = batch.to(device)
        batch.pos = batch.pos.clone().detach().requires_grad_(True)
        pred = compute_batch_predictions(batch, nnp, baseline_params, create_graph=True)

        n_atoms_graph = pred["n_atoms_graph"]
        E_per_atom_true  = pred["E_tot_true"] / n_atoms_graph
        mu_true    = batch.dipole.view(batch.num_graphs, 3)
        alpha_true = batch.polarizability.view(batch.num_graphs, 3, 3)

        # ---- L(E, E_tot): Eq. (4) with E_tot, plus L_aux of Eq. (5) ----
        energy_tot_graph = (pred["E_tot_pred"] / n_atoms_graph - E_per_atom_true) ** 2
        force_tot_graph  = atom_mse_to_graph((pred["forces_pred"] - batch.forces) ** 2, batch.batch, batch.num_graphs)
        mu_tot_graph     = ((pred["mu_pred"] - mu_true) ** 2).mean(dim=1)
        alpha_tot_graph  = ((pred["alpha_pred"] - alpha_true) ** 2).mean(dim=(1, 2))
        charge_graph     = atom_mse_to_graph((pred["Q_batch"] - batch.charges) ** 2, batch.batch, batch.num_graphs)
        if model_type != "Qeq":
            dipole_atom_graph = atom_mse_to_graph((pred["P_batch"] - batch.dipoles) ** 2, batch.batch, batch.num_graphs)
        else:
            dipole_atom_graph = torch.zeros(batch.num_graphs, dtype=dtype, device=device)

        # ---- L(E, E_b): Eq. (4) evaluated with the baseline alone (4.7-7.1 Å dimers only) ----
        energy_b_graph = (pred["E_b_pred"] / n_atoms_graph - E_per_atom_true) ** 2
        force_b_graph  = atom_mse_to_graph((pred["forces_b_pred"] - batch.forces) ** 2, batch.batch, batch.num_graphs)
        mu_b_graph     = ((pred["mu_b_pred"] - mu_true) ** 2).mean(dim=1)
        alpha_b_graph  = ((pred["alpha_b_pred"] - alpha_true) ** 2).mean(dim=(1, 2))

        loss_tot_graph = w_E * energy_tot_graph + w_F * force_tot_graph + w_q * charge_graph + w_p * dipole_atom_graph + w_mu * mu_tot_graph + w_alpha * alpha_tot_graph
        loss_b_graph   = w_E * energy_b_graph   + w_F * force_b_graph   + w_mu * mu_b_graph + w_alpha * alpha_b_graph

        mode = batch.train_loss_mode.view(-1).to(device=device)
        loss_graph = loss_tot_graph + torch.where(mode == 1, loss_b_graph, torch.zeros_like(loss_b_graph))
        loss = loss_graph.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_E = energy_tot_graph.mean()
        loss_F = force_tot_graph.mean()
        diff_E = (torch.abs(pred["E_tot_pred"] - pred["E_tot_true"])).sum() / batch.species.shape[0]
        diff_F = torch.abs(pred["forces_pred"] - batch.forces).mean()

        total_loss_sum += loss.item()
        loss_E_sum     += loss_E.item()
        loss_F_sum     += loss_F.item()
        diff_E_sum     += diff_E.item()
        diff_F_sum     += diff_F.item()
        num_batches += 1

        print(
            f"Epoch {epoch} "
            f"Loss_E: {loss_E.item():.6f}, "
            f"Loss_F: {loss_F.item():.6f}, "
            f"Loss_q: {charge_graph.mean().item():.6f}, "
            f"Loss_mu: {mu_tot_graph.mean().item():.6f} "
            f"Loss_alpha: {alpha_tot_graph.mean().item():.6f} "
            f"|dE| (meV/atom): {diff_E.item()*Ha2mev:.6f}, "
            f"|dF| (meV/Å): {diff_F.item()*Ha_Bohr2meV_angstrom:.6f}"
        )

    history[0].append(total_loss_sum / num_batches)
    history[1].append(loss_E_sum / num_batches)
    history[2].append(loss_F_sum / num_batches)
    history[3].append(diff_E_sum / num_batches)
    history[4].append(diff_F_sum / num_batches)

    print(
        f"Epoch {epoch:03d} | "
        f"Loss_E: {history[1][-1]:.6f}, "
        f"Loss_F: {history[2][-1]:.6f}, "
        f"|dE|: {history[3][-1]*Ha2mev:.6f} | "
        f"|dF|: {history[4][-1]*Ha_Bohr2meV_angstrom:.6f}"
    )

    # ---- every 10 epochs: RMSEs on the training dimers for checkpoint selection (S_4, SI Note 3) ----
    if epoch == 0 or (epoch + 1) % 10 == 0:
        nnp.eval()
        baseline_params.eval()
        optimizer.zero_grad(set_to_none=True)
        validation_values = evaluate_validation_batch(validation_loader, "Validation", nnp, baseline_params)
        for metric_list, value in zip(validation_metric, validation_values):
            metric_list.append(value)
        nnp.train()
        baseline_params.train()

        torch.save({
            'epoch': epoch,
            'nnp_state_dict': nnp.state_dict(),
            'dp_state_dict': baseline_params.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'validation_metric': validation_metric
        }, f"nnp_checkpoint_epoch_{epoch+1}.pt")

    torch.save({
        'epoch': epoch,
        'nnp_state_dict': nnp.state_dict(),
        'dp_state_dict': baseline_params.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'validation_metric': validation_metric
    }, checkpoint_path)

print("Energy RMSE (meV/atom):\n", validation_metric[0])
print("Force  RMSE (meV/Å):\n", validation_metric[1])
print("Mu     RMSE (e Bohr):\n", validation_metric[2])
print("Alpha  RMSE (Bohr^3):\n", validation_metric[3])
print("R2:\n", validation_metric[4])
