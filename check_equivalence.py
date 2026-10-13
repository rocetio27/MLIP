"""Check that the cleaned code reproduces the original code bit-for-bit (up to float32 round-off).

Usage:  python check_equivalence.py <dir_original> <dir_cleaned> <checkpoint.pt>
Runs each code base in a fresh interpreter on the first training dimer and monomer,
then compares E_tot, forces, mu, alpha, q and the baseline energy.
"""
import json, subprocess, sys, textwrap

ORIG, NEW, CKPT = sys.argv[1], sys.argv[2], sys.argv[3]

PROBE = textwrap.dedent("""
import sys, json, copy, torch
sys.path.insert(0, sys.argv[1])
import module as cfg
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph
import neural_networks as NN
import train_helper as TH
torch.manual_seed(0)
nnp = NN.NeuralNetworkPotential().to(device=cfg.device, dtype=cfg.dtype)
base = (NN.BaselineParameters if hasattr(NN, 'BaselineParameters') else NN.EnvDependentParamNet)().to(device=cfg.device, dtype=cfg.dtype)
ck = torch.load(sys.argv[2], map_location=cfg.device, weights_only=False)
sd = {k.replace('radial_basis.bpi', 'radial_basis.kappa') if hasattr(NN, 'BaselineParameters') else k: v
      for k, v in ck['nnp_state_dict'].items() if not (hasattr(NN, 'BaselineParameters') and k.startswith('node_embedding.muq_lin.'))}
nnp.load_state_dict(sd); base.load_state_dict(ck['dp_state_dict'])
nnp.eval(); base.eval()
graphs = []
for name in [cfg.dataset_key1, cfg.dataset_key2]:
    s = torch.load(name, map_location='cpu', weights_only=False)[0]
    pos = s.pos.to(device=cfg.device, dtype=cfg.dtype); Z = s.species.to(cfg.device)
    c = (Z[:, None] * pos).sum(0, keepdim=True) / Z.sum(); pos = pos - c
    graphs.append(Data(pos=pos, species=Z, E_tot=s.E_tot.to(cfg.device), charge_tot=s.charge_tot.to(cfg.device),
                       edge_index=radius_graph(pos, r=cfg.r_cut, loop=False)))
b = Batch.from_data_list(graphs).to(cfg.device); b.pos = b.pos.detach().requires_grad_(True)
p = TH.compute_batch_predictions(b, nnp, base, create_graph=False)
g = lambda *ks: next(p[k] for k in ks if k in p)
out = dict(E=g('E_tot_pred'), Eb=g('E_b_pred', 'E_pred_lr'), F=g('forces_pred'), mu=g('mu_pred', 'dipole_pred'),
           alpha=g('alpha_pred'), q=g('Q_batch'))
print(json.dumps({k: v.detach().cpu().double().flatten().tolist() for k, v in out.items()}))
""")

def run(d):
    r = subprocess.run([sys.executable, "-c", PROBE, d, CKPT], capture_output=True, text=True, cwd=d)
    if r.returncode != 0:
        sys.exit(f"--- failed in {d} ---\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])

a, b = run(ORIG), run(NEW)
ok = True
for k in a:
    import math
    diff = max(abs(x - y) for x, y in zip(a[k], b[k]))
    scale = max(1e-12, max(abs(x) for x in a[k]))
    print(f"{k:6s} max|diff| = {diff:.3e}   (relative {diff/scale:.1e})")
    ok &= diff / scale < 1e-5
print("EQUIVALENT" if ok else "MISMATCH")
