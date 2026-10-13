import math

import torch
from sklearn.metrics import r2_score
from torch_scatter import scatter_add

from module import ANG2BOHR, Ha2mev, Ha_Bohr2meV_angstrom, Z2onehotidx, device, dtype, model_type, onehotidx2atomE_tot, r_cut, polynomial_cutoff_p
from utils import compute_electrostatics


def atom_mse_to_graph(values, batch_index, num_graphs):
    """Atom-wise squared errors -> one mean per graph, i.e. the (1/N) sum_i ||.|| terms of Eq. (4)."""
    atom_mse = values.mean(dim=-1) if values.dim() > 1 else values
    graph_sum = scatter_add(atom_mse, batch_index, dim=0, dim_size=num_graphs)
    graph_count = scatter_add(torch.ones_like(atom_mse, dtype=dtype), batch_index, dim=0, dim_size=num_graphs)
    return graph_sum / graph_count.clamp_min(1.0)


def compute_batch_predictions(batch, nnp, baseline_params, create_graph=True):
    """Solve the baseline (Eq. (3)) graph by graph, evaluate the residual network, and return
    E_tot = E_b + E_nn together with its derivatives:
    F = -dE/dr, mu = -dE/d eps, alpha = -d^2E/d eps^2 (all at eps = 0)."""
    E_vec = torch.zeros(batch.num_graphs, 3, dtype=dtype, device=device, requires_grad=True)
    if model_type == "Qeq":
        chi, J, Rq = baseline_params(batch)
        inv_alpha = Rp = None
    else:
        chi, J, Rq, inv_alpha, Rp = baseline_params(batch)

    E_b_list, Q_list, P_list, v_list = [], [], [], []
    for i in range(batch.num_graphs):
        i_mask = (batch.batch == i)
        args = (batch.pos[i_mask], chi[i_mask], J[i_mask], Rq[i_mask],
                inv_alpha[i_mask] if inv_alpha is not None else None,
                Rp[i_mask] if Rp is not None else None,
                batch.charge_tot[i], E_vec[i], model_type, r_cut, polynomial_cutoff_p)
        if model_type == "Qeq":
            Q, E_b, v_potential = compute_electrostatics(*args)
        else:
            Q, P, E_b, v_potential = compute_electrostatics(*args)
            P_list.append(P.reshape(3, -1).T.contiguous())
        E_b_list.append(E_b)
        Q_list.append(Q)
        v_list.append(v_potential)

    Q_batch = torch.cat(Q_list, dim=0)
    batch.charges_pred = Q_batch
    batch.v_potential_pred = torch.cat(v_list, dim=0)
    P_batch = None
    if model_type != "Qeq":
        P_batch = torch.cat(P_list, dim=0)
        batch.dipoles_pred = P_batch

    E_nn_pred = scatter_add(nnp(batch, E_vec).squeeze(-1), batch.batch, dim=0, dim_size=batch.num_graphs)
    E_b_pred  = torch.stack(E_b_list).view(-1) + baseline_params.energy_offset_graphs.view(-1)   # E_b incl. sum_i E_i^0
    E_tot_pred = E_nn_pred + E_b_pred

    mu_nn_pred = -torch.autograd.grad(E_nn_pred.sum(), E_vec, create_graph=True, retain_graph=True)[0]
    mu_b_pred  = -torch.autograd.grad(E_b_pred.sum(),  E_vec, create_graph=True, retain_graph=True)[0]
    mu_pred = mu_nn_pred + mu_b_pred

    alpha_rows_nn, alpha_rows_b = [], []
    for a in range(3):
        alpha_rows_nn.append(torch.autograd.grad(mu_nn_pred[:, a].sum(), E_vec, create_graph=create_graph, retain_graph=True)[0])
        alpha_rows_b.append(torch.autograd.grad(mu_b_pred[:, a].sum(),  E_vec, create_graph=create_graph, retain_graph=True)[0])
    alpha_b_pred = torch.stack(alpha_rows_b, dim=1)
    alpha_pred   = torch.stack(alpha_rows_nn, dim=1) + alpha_b_pred

    forces_b_pred = -torch.autograd.grad(E_b_pred.sum(),   batch.pos, create_graph=create_graph, retain_graph=True)[0]
    forces_pred   = -torch.autograd.grad(E_tot_pred.sum(), batch.pos, create_graph=create_graph, retain_graph=True)[0]

    E_tot_true = batch.E_tot.view(-1) - scatter_add(onehotidx2atomE_tot[Z2onehotidx[batch.species]], batch.batch, dim=0, dim_size=batch.num_graphs)
    n_atoms_graph = scatter_add(torch.ones_like(batch.batch, dtype=dtype), batch.batch, dim=0, dim_size=batch.num_graphs)

    return {
        "Q_batch": Q_batch,
        "P_batch": P_batch,
        "E_tot_pred": E_tot_pred,
        "E_b_pred": E_b_pred,
        "E_tot_true": E_tot_true,
        "n_atoms_graph": n_atoms_graph,
        "mu_pred": mu_pred,
        "mu_b_pred": mu_b_pred,
        "alpha_pred": alpha_pred,
        "alpha_b_pred": alpha_b_pred,
        "forces_pred": forces_pred,
        "forces_b_pred": forces_b_pred,
    }


def evaluate_validation_batch(loader, split_name, nnp, baseline_params):
    """RMSEs used for checkpoint selection (SI Note 3). Returns [RMSE_E, RMSE_F, RMSE_mu, RMSE_alpha, R2]."""
    E_pred_list, E_true_list = [], []
    E_err_list, F_err_list, Mu_err_list, A_err_list = [], [], [], []
    N_sample = 0
    N_atom_all_samples = 0
    for batch in loader:
        batch = batch.to(device)
        batch.pos = batch.pos.clone().detach().requires_grad_(True)
        pred = compute_batch_predictions(batch, nnp, baseline_params, create_graph=False)
        mu_true = batch.dipole.view(batch.num_graphs, 3)
        alpha_true = batch.polarizability.view(batch.num_graphs, 3, 3)
        E_err_abs = torch.abs(pred["E_tot_pred"] - pred["E_tot_true"])
        F_err_atom = torch.abs(pred["forces_pred"] - batch.forces)
        Mu_err_abs = torch.abs(pred["mu_pred"] - mu_true)
        A_err_abs = torch.abs(pred["alpha_pred"] - alpha_true)
        for i in range(batch.num_graphs):
            atom_mask = (batch.batch == i)
            E_i = E_err_abs[i] * Ha2mev / pred["n_atoms_graph"][i]
            F_i = F_err_atom[atom_mask].mean() * Ha_Bohr2meV_angstrom
            Mu_i = Mu_err_abs[i].mean()
            A_i = A_err_abs[i].mean()
            d_i = batch.d.view(-1)[i] / ANG2BOHR
            print(f"|dE| (meV/atom): {E_i.item():.6f}, |dF| (meV/Å): {F_i.item():.6f}, |dMu| (e Bohr): {Mu_i.item():.6f}, |dA| (Bohr^3): {A_i.item():.6f}, |R| (Å): {d_i.item():.6f}")
        E_err_list.append((E_err_abs / pred["n_atoms_graph"]).detach())
        F_err_list.append(F_err_atom.flatten().detach())
        Mu_err_list.append(Mu_err_abs.flatten().detach())
        A_err_list.append(A_err_abs.flatten().detach())
        N_sample += batch.num_graphs
        N_atom_all_samples += batch.species.shape[0]
        E_pred_list.extend((pred["E_tot_pred"].detach() * Ha2mev / 1000).cpu().tolist())
        E_true_list.extend((pred["E_tot_true"].detach() * Ha2mev / 1000).cpu().tolist())
    E_err_tensor = torch.cat(E_err_list, dim=-1)
    F_err_tensor = torch.cat(F_err_list, dim=-1)
    Mu_err_tensor = torch.cat(Mu_err_list, dim=-1)
    A_err_tensor = torch.cat(A_err_list, dim=-1)
    RMSE_E = math.sqrt((E_err_tensor**2).sum().item() / N_sample) * Ha2mev
    RMSE_F = math.sqrt((F_err_tensor**2).sum().item() / (N_atom_all_samples * 3.0)) * Ha_Bohr2meV_angstrom
    RMSE_Mu = math.sqrt((Mu_err_tensor**2).mean().item())
    RMSE_A = math.sqrt((A_err_tensor**2).mean().item())
    R2 = r2_score(E_true_list, E_pred_list) if len(E_true_list) > 1 else float("nan")
    print("==============================================================")
    print(f"[{split_name} Results]")
    print(f"Energy RMSE : {RMSE_E:.6f} (meV/atom)")
    print(f"Force  RMSE : {RMSE_F:.6f} (meV/Å)")
    print(f"Mu     RMSE : {RMSE_Mu:.6f} (e Bohr)")
    print(f"Alpha  RMSE : {RMSE_A:.6f} (Bohr^3)")
    print(f"R2 = {R2:.4f}")
    print("==============================================================")
    return [RMSE_E, RMSE_F, RMSE_Mu, RMSE_A, R2]
