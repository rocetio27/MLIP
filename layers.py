import torch
import torch.nn as nn
from e3nn import o3
from e3nn import nn as o3nn


class SelfInteraction(nn.Module):
    def __init__(self, irreps_in, mul_out, act_0e=False):
        super().__init__()
        self.irreps_in = irreps_in
        self.mul_out = mul_out
        self.act_0e = act_0e

        irreps_list_out = [(mul_out[ir.l], (ir.l, ir.p)) for _, ir in self.irreps_in]
        self.irreps_out = o3.Irreps(irreps_list_out)

        self.si = o3.Linear(self.irreps_in, self.irreps_out, internal_weights=True, shared_weights=True, biases=True)

        if self.act_0e:
            acts = []
            for mul, ir in self.irreps_out:
                if ir.l == 0 and ir.p == 1:
                    acts.append(torch.nn.functional.silu)
                else:
                    acts.append(None)
            self.act = o3nn.Activation(self.irreps_out, acts)
            self.irreps_out = self.act.irreps_out
        else:
            self.act = None

    def forward(self, irrepsF):
        out = self.si(irrepsF)
        if self.act is not None:
            out = self.act(out)
        return out


class AtomWiseSelfInteraction(nn.Module):
    def __init__(self, Z2onehotidx_dict, irreps_in, mul_out):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.mul_out = mul_out
        self.Z2onehotidx_dict = Z2onehotidx_dict
        self.N_spec = len(Z2onehotidx_dict)

        irreps_list_out = [(mul_out[ir.l], ir) for mul, ir in self.irreps_in]
        self.irreps_out = o3.Irreps(irreps_list_out)

        self.linears = nn.ModuleList([
            o3.Linear(
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_out,
                internal_weights=True,
                shared_weights=True,
                biases=True
            )
            for _ in range(self.N_spec)
        ])

        total_params = sum(p.numel() for lin in self.linears for p in lin.parameters())
        print(f"[Params][AtomWiseSelfInteraction] = {total_params:,}")

    def forward(self, species, irrepsF):
        batch_dim = irrepsF.shape[0]
        out = torch.zeros(
            batch_dim,
            self.irreps_out.dim,
            device=irrepsF.device,
            dtype=irrepsF.dtype
        )

        for spec, onehot_idx in self.Z2onehotidx_dict.items():
            mask = (species == spec)
            if not mask.any():
                continue
            out[mask] = self.linears[onehot_idx](irrepsF[mask])

        return out
