from layers import SelfInteraction
from blocks import InteractionBlock
from graph_embedding import NodeEmbedding_C, NodeEmbedding_CD
from torch_scatter import scatter_add
from utils import compute_sh_all, polynomial_cutoff
import torch.nn as nn
import torch
import torch.nn.functional as F
from module import (
    mul,
    mul_end,
    lf_max,
    allowed_irreps,
    Z2onehotidx,
    Z2onehotidx_dict,
    Y_irreps,
    model_type,
    r_cut,
    polynomial_cutoff_p,
)


class BaselineParameters(nn.Module):
    """Element-specific, optimisable parameters of the electrostatic baseline, Eq. (1):
    E_i^0 (energy_offset), chi_i, J_i, R_q^i, alpha_i (stored as 1/alpha_i), R_p^i."""
    def __init__(self):
        super().__init__()
        self.n_species = len(Z2onehotidx_dict)

        self.chi_embed = nn.Embedding(self.n_species, 1)   # electronegativity chi_i
        self.J_embed   = nn.Embedding(self.n_species, 1)   # hardness J_i (softplus > 0)
        self.Rq_embed  = nn.Embedding(self.n_species, 1)   # charge Gaussian width R_q^i (softplus > 0)

        # E_i^0: species-wise reference atomic energy (Hartree / atom), added on top of the
        # fixed isolated-atom energies subtracted from the labels (module.onehotidx2atomE_tot)
        self.energy_offset_embed = nn.Embedding(self.n_species, 1)
        nn.init.zeros_(self.energy_offset_embed.weight)

        if model_type in ["CDeq", "DDeq"]:
            self.K_embed  = nn.Embedding(self.n_species, 1)  # 1/alpha_i, inverse atomic polarizability (softplus > 0)
            self.Rp_embed = nn.Embedding(self.n_species, 1)  # dipole Gaussian width R_p^i (softplus > 0)

    def forward(self, batch):
        idx = Z2onehotidx[batch.species]  # [Na]

        chi = self.chi_embed(idx)[:, 0]
        J   = F.softplus(self.J_embed(idx)[:, 0])
        Rq  = F.softplus(self.Rq_embed(idx)[:, 0])

        self.energy_offset_atoms  = self.energy_offset_embed(idx)[:, 0]          # [Na]
        self.energy_offset_graphs = scatter_add(self.energy_offset_atoms, batch.batch, dim=0)  # [num_graphs]

        if model_type == "Qeq":
            return chi, J, Rq

        inv_alpha = F.softplus(self.K_embed(idx)).expand(-1, 3)   # [Na, 3]  1/alpha_i (isotropic)
        Rp        = F.softplus(self.Rp_embed(idx)[:, 0])          # [Na]
        return chi, J, Rq, inv_alpha, Rp


class NeuralNetworkPotential(nn.Module):
    """Residual energy E_nn: node embedding -> 4 interaction blocks -> per-atom scalar readout."""
    def __init__(self):
        super().__init__()
        if model_type == "Qeq":
            self.node_embedding = NodeEmbedding_C()
        else:
            self.node_embedding = NodeEmbedding_CD()

        self.int_block1 = InteractionBlock(Y_irreps, self.node_embedding.irreps_out, allowed_irreps, mul, Z2onehotidx_dict)
        self.int_block2 = InteractionBlock(Y_irreps, self.int_block1.irreps_out, allowed_irreps, mul, Z2onehotidx_dict)
        self.int_block3 = InteractionBlock(Y_irreps, self.int_block2.irreps_out, allowed_irreps, mul, Z2onehotidx_dict)
        self.int_block4 = InteractionBlock(Y_irreps, self.int_block3.irreps_out, allowed_irreps, mul, Z2onehotidx_dict)
        self.si = SelfInteraction(self.int_block4.irreps_out, mul_end)

    def forward(self, batch, E_vec):
        N_a = batch.species.shape[0]
        species = batch.species
        src, dst = batch.edge_index
        edge_batch = batch.batch[src]

        charges     = batch.charges_pred           # q_i from the baseline, Eq. (3)
        v_potential = batch.v_potential_pred       # [v_i^in, v_i^out], Eq. (S8)

        dist_vecs = batch.pos[src] - batch.pos[dst]
        edge_dist = torch.norm(dist_vecs, dim=-1)
        Y = compute_sh_all(dist_vecs / (edge_dist[:, None] + 1e-9), lf_max)

        # effective neighbour count N_i = sum_j f_env(r_ij / r_c), Eq. (S16)
        f_env = polynomial_cutoff(edge_dist / r_cut, polynomial_cutoff_p)
        N_i = scatter_add(f_env, src, dim=0, dim_size=N_a)

        if model_type == "Qeq":
            irrepsF = self.node_embedding(species, charges, E_vec, v_potential, batch.batch)
        else:
            irrepsF = self.node_embedding(species, charges, batch.dipoles_pred, E_vec, v_potential, batch.batch)

        irrepsF = self.int_block1(Y, edge_dist, dist_vecs, edge_batch, E_vec, src, dst, species, irrepsF, N_i, N_a)
        irrepsF = self.int_block2(Y, edge_dist, dist_vecs, edge_batch, E_vec, src, dst, species, irrepsF, N_i, N_a)
        irrepsF = self.int_block3(Y, edge_dist, dist_vecs, edge_batch, E_vec, src, dst, species, irrepsF, N_i, N_a)
        irrepsF = self.int_block4(Y, edge_dist, dist_vecs, edge_batch, E_vec, src, dst, species, irrepsF, N_i, N_a)
        irrepsF_out = self.si(irrepsF)

        # per-atom residual energy: first channel of the 0e block of the readout
        slices = self.si.irreps_out.slices()
        E_nn_atoms = irrepsF_out[:, slices[1]][:, 0]
        return E_nn_atoms.unsqueeze(-1)
