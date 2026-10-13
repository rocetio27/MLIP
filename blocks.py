import torch
import torch.nn as nn
from layers import SelfInteraction, AtomWiseSelfInteraction
from e3nn import o3
from utils import KronAdd
from torch_scatter import scatter_add
from graph_embedding import RadialMLP
from module import norm_eps


class InteractionBlock(nn.Module):
    """One NequIP-style interaction layer.
    Message  m_ij^(k) = R_ij^(k) * [ Y(r_hat_ij) (x)_CG V_j^(k-1) ]                 Eq. (S14)
    Aggregate (1/sqrt(N_i)) sum_{j in N(i)} m_ij^(k),  N_i = sum_j f_env(r_ij/r_c)  Eqs. (S15)-(S16)
    """
    def __init__(self, Y_irreps, irreps_in, allowed_irreps, mul, Z2onehotidx_dict):
        super().__init__()
        self.ftp           = o3.FullTensorProduct(irreps_in1=Y_irreps, irreps_in2=irreps_in, filter_ir_out=allowed_irreps)
        self.radial_mlp    = RadialMLP(self.ftp.irreps_out.simplify())
        self.si_conv       = SelfInteraction(self.ftp.irreps_out.simplify(), mul)
        self.awsi_residual = AtomWiseSelfInteraction(Z2onehotidx_dict, irreps_in, mul)
        self.kron_add      = KronAdd(X_irreps=self.si_conv.irreps_out, Y_irreps=self.awsi_residual.irreps_out, allowed_irreps=allowed_irreps)
        self.si_kron_add   = SelfInteraction(self.kron_add.irreps_out, mul, act_0e=True)
        self.irreps_out    = self.kron_add.irreps_out

    def forward(self, Y, edge_dist, dist_vecs, edge_batch, E_vec, src, dst, species, irrepsF_in, N_i, N_a):
        V_j      = irrepsF_in[dst]
        T        = self.ftp(Y, V_j)
        R        = self.radial_mlp(edge_dist, dist_vecs, E_vec, edge_batch)
        # node-wise normalisation 1/sqrt(N_i + eps), Eq. (S15)
        messages = T * R / torch.sqrt(N_i[src] + norm_eps).unsqueeze(-1)
        aggregated   = scatter_add(messages, src, dim=0, dim_size=N_a)
        conv         = self.si_conv(aggregated)
        residual     = self.awsi_residual(species, irrepsF_in)
        irrepsF_out  = self.si_kron_add(self.kron_add(conv, residual))
        return irrepsF_out
