from e3nn import o3
import torch
import torch.nn as nn
import math
from layers import SelfInteraction
from utils import polynomial_cutoff
from module import Z2onehotidx, N_r, mul, N_spec, N_b, r_cut, hidden_dim, polynomial_cutoff_p


class NodeEmbedding_C(nn.Module):
    """Node embedding for the Qeq baseline (charges only).
    0e: Z_i, q_i, |eps|^2, v_i^in, v_i^out ;  1o: eps
    """
    def __init__(self):
        super().__init__()
        self.embedder = nn.Embedding(N_spec, N_r)
        self.charge_lin = nn.Linear(1, N_r, bias=False)
        self.E2_lin = nn.Linear(1, N_r, bias=False)
        self.charge_potential_in_lin = nn.Linear(1, N_r, bias=False)    # v_i^in
        self.charge_potential_out_lin = nn.Linear(1, N_r, bias=False)   # v_i^out

        self.irreps_1o_in = o3.Irreps("1x1o")
        self.irreps_1o_out = o3.Irreps(f"{N_r}x1o")
        self.efield_lin = o3.Linear(self.irreps_1o_in, self.irreps_1o_out, internal_weights=True, shared_weights=True, biases=False)

        self.irreps_in = o3.Irreps(f"{5 * N_r}x0e + {1 * N_r}x1o")
        self.si_ZC = SelfInteraction(self.irreps_in, mul)
        self.irreps_out = self.si_ZC.irreps_out

    def forward(self, species, charges, E_vec, v_potential, node_batch):
        node2onehotidx = Z2onehotidx[species]

        q = charges.view(-1, 1)
        E_node = E_vec[node_batch]
        E2 = (E_node**2).sum(dim=-1, keepdim=True)
        v_in = v_potential[:, 0:1]
        v_out = v_potential[:, 1:2]

        irrepsF_0e = torch.cat([self.embedder(node2onehotidx), self.charge_lin(q), self.E2_lin(E2),
                                self.charge_potential_in_lin(v_in), self.charge_potential_out_lin(v_out)], dim=-1)
        irrepsF_1o = self.efield_lin(E_node)
        return self.si_ZC(torch.cat([irrepsF_0e, irrepsF_1o], dim=-1))


class NodeEmbedding_CD(nn.Module):
    """Node embedding for the charge-dipole baselines (CDeq, DDeq), Eq. (S7).
    0e: Z_i, q_i, |eps|^2, p_i . eps, v_i^in, v_i^out ;  1o: eps, p_i
    """
    def __init__(self):
        super().__init__()
        self.embedder = nn.Embedding(N_spec, N_r)
        self.charge_lin = nn.Linear(1, N_r, bias=False)
        self.E2_lin = nn.Linear(1, N_r, bias=False)
        self.p_dot_E_lin = nn.Linear(1, N_r, bias=False)
        self.charge_potential_in_lin = nn.Linear(1, N_r, bias=False)    # v_i^in
        self.charge_potential_out_lin = nn.Linear(1, N_r, bias=False)   # v_i^out

        self.irreps_1o_in = o3.Irreps("1x1o")
        self.irreps_1o_out = o3.Irreps(f"{N_r}x1o")
        self.efield_lin = o3.Linear(self.irreps_1o_in, self.irreps_1o_out, internal_weights=True, shared_weights=True, biases=False)
        self.dipole_lin = o3.Linear(self.irreps_1o_in, self.irreps_1o_out, internal_weights=True, shared_weights=True, biases=False)

        self.irreps_in = o3.Irreps(f"{6 * N_r}x0e + {2 * N_r}x1o")
        self.si_ZCD = SelfInteraction(self.irreps_in, mul)
        self.irreps_out = self.si_ZCD.irreps_out

    def forward(self, species, charges, dipoles, E_vec, v_potential, node_batch):
        node2onehotidx = Z2onehotidx[species]

        q = charges.view(-1, 1)
        E_node = E_vec[node_batch]
        E2 = (E_node**2).sum(dim=-1, keepdim=True)
        p_dot_E = (dipoles * E_node).sum(dim=-1, keepdim=True)
        v_in = v_potential[:, 0:1]
        v_out = v_potential[:, 1:2]

        irrepsF_0e = torch.cat([self.embedder(node2onehotidx), self.charge_lin(q), self.E2_lin(E2), self.p_dot_E_lin(p_dot_E),
                                self.charge_potential_in_lin(v_in), self.charge_potential_out_lin(v_out)], dim=-1)
        irrepsF_1o = torch.cat([self.efield_lin(E_node), self.dipole_lin(dipoles)], dim=-1)
        return self.si_ZCD(torch.cat([irrepsF_0e, irrepsF_1o], dim=-1))


class RadialBasis(torch.nn.Module):
    """Bessel basis B_n(r) = (2/r_c) sin(kappa_n r / r_c) / r, Eq. (S12); kappa_n initialised at n*pi and trainable."""
    def __init__(self, r_cut):
        super().__init__()
        self.r_cut = r_cut
        self.kappa = torch.nn.Parameter(torch.arange(1, N_b + 1) * math.pi)

    def forward(self, r):
        x = r / self.r_cut
        return (2.0 / self.r_cut) * torch.sin(self.kappa[None, :] * x[:, None]) / r[:, None]


class RadialMLP(nn.Module):
    """Radial coefficients R_ij^(k) = W3 silu(W2 silu(W1 X_ij)), Eq. (S13), with
    X_ij = f_env(r_ij/r_c) concat[B(r_ij), |eps|^2, r_hat_ij . eps], Eq. (S10)."""
    def __init__(self, irreps_in):
        super().__init__()
        self.pre_dim = 0
        iL_index = []
        for mul_i, ir in irreps_in:
            for _ in range(mul_i):
                self.pre_dim += 1
                iL_index.append(ir.l)

        self.register_buffer("iL_index", torch.tensor(iL_index, dtype=torch.long))
        self.radial_basis = RadialBasis(r_cut)
        self.mlp = nn.Sequential(
            nn.Linear(N_b + 2, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.pre_dim, bias=False),
        )

        total_params = sum(p.numel() for p in self.mlp.parameters())
        print(f"[Params][RadialMLP]: {total_params:,}")

    def forward(self, r, dist_vecs, E_vec, edge_batch):
        E_edge = E_vec[edge_batch, :]
        f_env = polynomial_cutoff(r / r_cut, polynomial_cutoff_p)
        B = self.radial_basis(r)
        E2 = (E_edge**2).sum(dim=-1)
        r_hat_dot_E = (dist_vecs * E_edge).sum(dim=-1) / torch.clamp(r, min=1e-12)
        X = f_env[:, None] * torch.cat([B, E2[:, None], r_hat_dot_E[:, None]], dim=-1)
        R = self.mlp(X)
        repeat_counts = 2 * self.iL_index + 1
        idxs = torch.arange(self.pre_dim, device=r.device).repeat_interleave(repeat_counts)
        return R[:, idxs]
