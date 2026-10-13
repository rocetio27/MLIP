from e3nn import o3
from pathlib import Path
from utils import parity2chr
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

ANG2BOHR = 1.889726125
Ha2mev = 27211.386245988
Ha_Bohr2meV_angstrom = 51422.067

# Isolated-atom reference energies (Hartree), subtracted from the DFT total energies.
onehotidx2atomE_tot = torch.tensor([-0.5010619187567114, -37.80543577171177, -54.54388843569469, -75.01925656374482], dtype=dtype, device=device)

Z2onehotidx_dict = {
    1: 0,  # H
    6: 1,  # C
    7: 2,  # N
    8: 3,  # O
}

max_Z = max(Z2onehotidx_dict.keys())
Z2onehotidx = torch.full((max_Z + 1,), -1, dtype=torch.long)
for Z, idx in Z2onehotidx_dict.items():
    Z2onehotidx[Z] = idx
Z2onehotidx = Z2onehotidx.to(device)
N_spec = len(Z2onehotidx_dict)

# ---- MPNN hyperparameters (SI Note 2) ----
polynomial_cutoff_p = 5                 # p in the envelope f_env, Eq. (S9)
r_cut       = 5.8 * ANG2BOHR            # message-passing cutoff r_c
N_r         = 8                         # width of each node-feature embedding
l_max       = 2                         # max l of node features
lf_max      = 2                         # max l of the spherical-harmonic edge embedding Y(r_hat_ij)
N_b         = 16                        # number of radial Bessel functions, Eq. (S11)
mul         = [32 for i in range(l_max + 1)]
mul_end     = [(5 if i == 0 else 1) for i in range(l_max + 1)]
hidden_dim  = 64                        # width of the radial MLP, Eq. (S13)
norm_eps    = 1e-8                      # 1/sqrt(N_i + eps) message normalisation, Eq. (S15)

Y_irreps       = o3.Irreps(" + ".join([f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lf_max + 1)]))
allowed_irreps = [o3.Irrep(l, p) for l in range(l_max + 1) for p in (-1, 1)]

# Resolve packaged data independently of the shell's working directory.
DATASET_DIR = Path(__file__).resolve().parent / "dataset" / "Organic ion-pair dimer"
dataset_key1 = str(DATASET_DIR / "dataset_ionic33_train_dimer.pt")
dataset_key2 = str(DATASET_DIR / "dataset_ionic33_train_monomer.pt")
model_type = "DDeq"  # Qeq, CDeq, DDeq (paper: DDeq = charge-dipole equilibration, Eq. (3))
