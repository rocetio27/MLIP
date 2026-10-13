import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import spherical_harmonics
import math


# =====================================================================================
#  Electrostatic baseline (main text Eqs. (1)-(3), SI Note 1)
#  Vector layout of atomic dipoles P: [p1x ... pNx, p1y ... pNy, p1z ... pNz]
# =====================================================================================
def compute_electrostatics(pos, chi, J, Rq, inv_alpha, Rp, Q_tot, E_vec, model_type, r_cut=None, p=5):
    Tqq = gen_Tqq(pos, J, Rq)                       # T_qq + J  (diagonal: sqrt(2/pi)/R_q + J)

    if model_type == "Qeq":
        Q, lam = solve_qeq(pos, chi, Q_tot, Tqq, E_vec)
        E_b = cal_baseline_energy(pos, chi, Q, None, Tqq, None, None, None, E_vec, model_type)
        v_potential = cal_charge_potential(pos, Tqq, Q, r_cut, p)
        return Q, E_b, v_potential

    Tpq     = gen_Tpq(pos, Rq, Rp)
    Tpp_dia = gen_Tpp_diagonal(inv_alpha, Rp)       # -T_pp^ii + 1/alpha_i, stored with a minus sign (see below)
    if model_type == "CDeq":
        Tpp_off = None
        Q, P, lam = solve_cdeq(pos, chi, Q_tot, Tqq, Tpq, Tpp_dia, E_vec)
    elif model_type == "DDeq":
        Tpp_off = gen_Tpp_off(pos, Rp)
        Q, P, lam = solve_ddeq(pos, chi, Q_tot, Tqq, Tpq, Tpp_off, Tpp_dia, E_vec)
    else:
        raise ValueError(f"Unknown electrostatics model_type: {model_type}")

    E_b = cal_baseline_energy(pos, chi, Q, P, Tqq, Tpq, Tpp_dia, Tpp_off, E_vec, model_type)
    v_potential = cal_charge_potential(pos, Tqq, Q, r_cut, p)
    return Q, P, E_b, v_potential


def cal_charge_potential(pos, Tqq, Q, r_cut=None, p=5):
    """Electrostatic potential descriptor v_i = sum_{j != i} T_qq^ij q_j split into
    v_i^in (weighted by f_env) and v_i^out (weighted by 1 - f_env), Eq. (S8). Returns [Na, 2]."""
    Tqq_off = Tqq.clone()
    Tqq_off.fill_diagonal_(0.0)
    if r_cut is None:
        v_in = Tqq_off @ Q
        return torch.stack([v_in, torch.zeros_like(v_in)], dim=-1)

    rij = torch.linalg.norm(pos[:, None, :] - pos[None, :, :], dim=-1)
    f_env = polynomial_cutoff(rij / r_cut, p)
    v_in  = (Tqq_off * f_env) @ Q
    v_out = (Tqq_off * (1.0 - f_env)) @ Q
    return torch.stack([v_in, v_out], dim=-1)


def cal_baseline_energy(pos, chi, Q, P, Tqq, Tpq, Tpp_dia, Tpp_off, E_vec, model_type):
    """Baseline energy E_b of Eqs. (1)-(2) without the E_i^0 term (added in train_helper).
    V_i = -r_i . eps, so sum_i q_i V_i - sum_i p_i . eps = -mu . eps."""
    V = -(pos * E_vec[None, :]).sum(dim=1)
    Na = pos.shape[0]

    Eqq = 0.5 * (Q @ (Tqq @ Q))                       # 1/2 sum T_qq q q  +  1/2 sum J q^2

    if model_type == "Qeq":
        return Eqq + (Q * (V + chi)).sum()

    E_tiled = E_vec.repeat_interleave(Na)
    Epq     = P @ (Tpq @ Q)                           # sum p . T_pq q
    Epp_dia = 0.5 * (P * (Tpp_dia * P)).sum(dim=-1)   # -(1/2) sum p^2 / alpha  -(1/2) sum p . T_pp^ii p
    E_b     = Eqq - Epq - Epp_dia + (Q * (V + chi)).sum() - (P * E_tiled).sum()
    if model_type == "DDeq":
        E_b = E_b - 0.5 * (P @ (Tpp_off @ P))         # -(1/2) sum_{i != j} p_i . T_pp^ij p_j
    return E_b


# ---------------------------- linear solvers, Eq. (3) ----------------------------
def solve_qeq(pos, chi, Q_tot, Tqq, E_vec):
    V = -(pos * E_vec[None, :]).sum(dim=1)
    Na = pos.shape[0]
    M = torch.zeros((Na + 1, Na + 1), device=pos.device, dtype=pos.dtype)
    M[:Na, :Na] = Tqq
    M[:Na, Na] = 1
    M[Na, :Na] = 1
    rhs = torch.zeros((Na + 1,), device=pos.device, dtype=pos.dtype)
    rhs[:Na] = -chi - V
    rhs[Na] = Q_tot
    sol = torch.linalg.solve(M, rhs)
    return sol[:Na], sol[Na]


def solve_cdeq(pos, chi, Q_tot, Tqq, Tpq, Tpp_dia, E_vec):
    """Charge-dipole equilibration with diagonal dipole-dipole coupling only; dipoles eliminated analytically."""
    Na = pos.shape[0]
    E_tiled = E_vec.repeat_interleave(Na)
    Tpp_dia_inv = 1 / (Tpp_dia + 1e-8)
    temp_mat = Tpp_dia_inv.unsqueeze(-1) * Tpq
    Tqp = Tpq.T
    V = -(pos * E_vec[None, :]).sum(dim=1)

    M = torch.zeros((Na + 1, Na + 1), device=pos.device, dtype=pos.dtype)
    M[:Na, :Na] = Tqq + Tqp @ temp_mat
    M[:Na, Na] = 1
    M[Na, :Na] = 1
    rhs = torch.zeros((Na + 1,), device=pos.device, dtype=pos.dtype)
    rhs[:Na] = -chi - Tqp @ (Tpp_dia_inv * E_tiled) - V
    rhs[Na] = Q_tot

    sol = torch.linalg.solve(M, rhs)
    Q = sol[:Na]
    P = -Tpp_dia_inv * E_tiled - temp_mat @ Q
    return Q, P, sol[Na]


def solve_ddeq(pos, chi, Q_tot, Tqq, Tpq, Tpp_off, Tpp_dia, E_vec):
    """Full charge-dipole equilibration, Eq. (3):
    [[T_qq + J, -T_qp, 1], [-T_pq, -T_pp + A^-1, 0], [1, 0, 0]] (q, p, lambda) = (-chi - V, eps, Q_tot)."""
    Na = pos.shape[0]
    E_tiled = E_vec.repeat_interleave(Na)
    M = torch.zeros((4 * Na + 1, 4 * Na + 1), device=pos.device, dtype=pos.dtype)
    M[:Na, :Na] = Tqq
    M[:Na, Na:4 * Na] = -Tpq.T
    M[Na:4 * Na, :Na] = -Tpq
    M[Na:4 * Na, Na:4 * Na] = -Tpp_off
    M[range(Na, 4 * Na), range(Na, 4 * Na)] = -Tpp_dia
    M[:Na, 4 * Na] = 1
    M[4 * Na, :Na] = 1

    rhs = torch.zeros((4 * Na + 1,), device=pos.device, dtype=pos.dtype)
    V = -(pos * E_vec[None, :]).sum(dim=1)
    rhs[:Na] = -chi - V
    rhs[Na:4 * Na] = E_tiled
    rhs[4 * Na] = Q_tot

    sol = torch.linalg.solve(M, rhs)
    return sol[:Na], sol[Na:4 * Na], sol[4 * Na]


# ---------------------------- interaction kernels, Eqs. (S4)-(S6) ----------------------------
def gen_Tqq(pos, J, Rq):
    """T_qq^ij = erf(x_qq^ij)/r_ij, diagonal T_qq^ii = sqrt(2/pi)/R_q^i; hardness J_i added on the diagonal."""
    Na = pos.shape[0]
    rij = torch.linalg.norm(pos[:, None, :] - pos[None, :, :], dim=-1)
    Rqq = torch.sqrt(Rq[:, None]**2 + Rq[None, :]**2)
    x = rij / Rqq

    Tqq = torch.zeros_like(rij)
    mask = rij > 1e-12
    Tqq[mask] = torch.erf(x[mask]) / rij[mask]

    idx = torch.arange(Na, device=pos.device)
    Tqq[idx, idx] = 1.0 / (math.sqrt(math.pi / 2) * Rq) + J
    return Tqq


def gen_Tpq(pos, Rq, Rp):
    """T_pq^ij = r_hat_ij / r_ij^2 [erf(x) - 2/sqrt(pi) x exp(-x^2)], Eq. (S5); returned as (3Na, Na)."""
    Na = pos.shape[0]
    rij_vec  = pos[:, None, :] - pos[None, :, :]
    rij_norm = torch.linalg.norm(rij_vec, dim=-1)
    Rpq = torch.sqrt(Rp[:, None]**2 + Rq[None, :]**2)
    x = rij_norm / Rpq

    Tpq_scalar = torch.zeros_like(rij_norm)
    mask = rij_norm > 1e-12
    Tpq_scalar[mask] = (torch.erf(x[mask]) - 2.0 / math.sqrt(math.pi) * x[mask] * torch.exp(-x[mask]**2)) / rij_norm[mask]**3
    Tpq = (Tpq_scalar[:, :, None] * rij_vec).permute(2, 0, 1).reshape(3 * Na, Na)
    return Tpq


def gen_Tpp_diagonal(inv_alpha, Rp):
    """Diagonal of the dipole block, stored as T_pp^ii - 1/alpha_i so that -Tpp_dia = -T_pp^ii + 1/alpha_i.
    T_pp^ii = -sqrt(2/pi) / (3 R_p^3) I  (SI Note 1)."""
    Tpp_self = -1.0 / (3.7599942 * Rp**3)                     # (Na,)  3*sqrt(pi/2) = 3.7599942
    return Tpp_self.repeat(3) - inv_alpha.T.flatten()         # (3Na,)


def gen_Tpp_off(pos, Rp):
    """Off-diagonal dipole-dipole kernel T_pp^ij, Eq. (S6); returned as (3Na, 3Na)."""
    device, dtype = pos.device, pos.dtype
    Na = pos.shape[0]

    Rpp = torch.sqrt(Rp[:, None]**2 + Rp[None, :]**2)
    rij_vec = pos[None, :, :] - pos[:, None, :]
    rij_norm = torch.linalg.norm(rij_vec, dim=-1)

    mask_offdiag = ~torch.eye(Na, dtype=torch.bool, device=device)
    rij_safe = torch.where(mask_offdiag, rij_norm, torch.ones_like(rij_norm))

    x = rij_safe / Rpp
    rij_hat = rij_vec / rij_safe[:, :, None]
    outer = rij_hat[..., :, None] * rij_hat[..., None, :]
    I3 = torch.eye(3, dtype=dtype, device=device).view(1, 1, 3, 3)

    term1 = (torch.special.erf(x) - (2.0 / math.sqrt(math.pi)) * x * torch.exp(-x**2)) / rij_safe**3
    term2 = (4.0 / (math.sqrt(math.pi) * Rpp**3)) * torch.exp(-x**2)

    Tij = (3.0 * outer - I3) * term1[..., None, None] - outer * term2[..., None, None]
    Tij[~mask_offdiag] = 0.0
    return Tij.permute(2, 0, 3, 1).reshape(3 * Na, 3 * Na)


# =====================================================================================
#  MPNN helpers
# =====================================================================================
def kron_add_l(x, y, l):
    """Additive Kronecker product of two feature blocks of the same irrep (l, p):
    x: (N, c1*(2l+1)), y: (N, c2*(2l+1)) -> (N, c1*c2*(2l+1))."""
    N = x.shape[0]
    dim_l = 2 * l + 1
    x = x.view(N, -1, dim_l)
    y = y.view(N, -1, dim_l)
    z = x[:, :, None, :] + y[:, None, :, :]
    return z.reshape(N, -1)


class KronAdd(nn.Module):
    def __init__(self, X_irreps, Y_irreps, allowed_irreps):
        super().__init__()
        self.X_irreps = X_irreps
        self.X_ir2mul_dict = {ir: mul for mul, ir in X_irreps}
        self.X_ir2idx_dict = {ir: i for i, (mul, ir) in enumerate(X_irreps)}
        self.X_ir_list = list(self.X_ir2mul_dict.keys())

        self.Y_irreps = Y_irreps
        self.Y_ir2mul_dict = {ir: mul for mul, ir in Y_irreps}
        self.Y_ir2idx_dict = {ir: i for i, (mul, ir) in enumerate(Y_irreps)}
        self.Y_ir_list = list(self.Y_ir2mul_dict.keys())

        self.allowed_irreps = allowed_irreps
        self.X_slices = X_irreps.slices()
        self.Y_slices = Y_irreps.slices()

        irreps_out_list = []
        for ir in allowed_irreps:
            if (ir in self.X_ir_list) and (ir in self.Y_ir_list):
                mul_X = self.X_ir2mul_dict[ir]
                mul_Y = self.Y_ir2mul_dict[ir]
                mul = mul_X if mul_X == mul_Y else mul_X * mul_Y
                irreps_out_list.append((mul, ir))
            elif ir in self.X_ir_list:
                irreps_out_list.append((self.X_ir2mul_dict[ir], ir))
            elif ir in self.Y_ir_list:
                irreps_out_list.append((self.Y_ir2mul_dict[ir], ir))
        self.irreps_out = o3.Irreps(irreps_out_list)

    def forward(self, X_irrepsF, Y_irrepsF):
        out = []
        for ir in self.allowed_irreps:
            if (ir in self.X_ir_list) and (ir in self.Y_ir_list):
                xs = X_irrepsF[:, self.X_slices[self.X_ir2idx_dict[ir]]]
                ys = Y_irrepsF[:, self.Y_slices[self.Y_ir2idx_dict[ir]]]
                if self.X_ir2mul_dict[ir] == self.Y_ir2mul_dict[ir]:
                    out.append(xs + ys)
                else:
                    out.append(kron_add_l(xs, ys, ir.l))
            elif ir in self.X_ir_list:
                out.append(X_irrepsF[:, self.X_slices[self.X_ir2idx_dict[ir]]])
            elif ir in self.Y_ir_list:
                out.append(Y_irrepsF[:, self.Y_slices[self.Y_ir2idx_dict[ir]]])
        return torch.cat(out, dim=-1)


def polynomial_cutoff(x, p: int):
    """Envelope f_env(x), Eq. (S9): 1 at x = 0; value and first two derivatives vanish at x = 1; 0 for x > 1."""
    if not isinstance(p, int) or p < 1:
        raise ValueError(f"p must be a positive integer, got {p}")
    xp  = x**p
    xp1 = xp * x
    xp2 = xp1 * x
    out = 1.0 - 0.5 * (p + 1) * (p + 2) * xp + p * (p + 2) * xp1 - 0.5 * p * (p + 1) * xp2
    return torch.where(x <= 1.0, out, torch.zeros_like(x))


def compute_sh_all(edge_dir_vecs, l_max):
    """Real spherical harmonics Y(r_hat_ij) for l = 0 ... l_max."""
    return spherical_harmonics(list(range(l_max + 1)), edge_dir_vecs, normalize=True, normalization="integral")


def parity2chr(p: int) -> str:
    if p not in (+1, -1):
        raise ValueError(f"Invalid parity {p}. Must be +1 or -1.")
    return "e" if p == +1 else "o"
