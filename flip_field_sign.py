"""Convert the water and NMA dataset files to the physical field-sign convention.

The NaCl and organic ion-pair files already store the physical external field
(E_vec = -E_psi4, see dataset/README.md).  The water files were stored in the
Psi4 ``perturb_dipole`` input convention, and the NMA files kept FIREANN's
``External_field`` and ``POL`` conventions.  This script negates

  water molecule/*.pt : E_vec, field
  NMA/*.pt            : E_vec, polarizability, external_field_v_per_ang

so that every dataset satisfies  E(eps) = E0 - mu . eps - 1/2 eps . alpha . eps,
mu = -dE/d eps and alpha = -d2E/d eps2, which is the convention used by
utils.cal_baseline_energy and train_helper.compute_batch_predictions.

The archives are rewritten at the byte level (no torch needed): only the raw
storage records of the listed tensors are negated; the pickle and all other
records are copied unchanged, so torch.load returns the same Data objects with
the listed fields negated.  The script refuses to run twice on the same file
(a marker record ``<archive>/physical_sign_convention`` is added).

Usage (from the repository root):   python flip_field_sign.py
"""
import array
import io
import pickle
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "dataset"
TARGETS = {
    "water molecule/dataset_water_x_rotation_z_field_m001_pbe0.pt": ["E_vec", "field"],
    "water molecule/dataset_water_z_field_sweep_pbe0.pt": ["E_vec", "field"],
    "NMA/dataset_nma_train.pt": ["E_vec", "polarizability", "external_field_v_per_ang"],
    "NMA/dataset_nma_validation.pt": ["E_vec", "polarizability", "external_field_v_per_ang"],
    "NMA/dataset_nma_test.pt": ["E_vec", "polarizability", "external_field_v_per_ang"],
}
MARKER = "physical_sign_convention"
TYPECODE = {"FloatStorage": "f", "DoubleStorage": "d"}


class TensorRef:
    def __init__(self, key, storage_type, offset, size, stride):
        self.key, self.storage_type, self.offset, self.size, self.stride = key, storage_type, offset, tuple(size), tuple(stride)

    @property
    def numel(self):
        n = 1
        for s in self.size:
            n *= s
        return n


class Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state


class Data(Stub):
    def __setstate__(self, state):
        store = state.get("_store")
        self.fields = dict(store.mapping) if isinstance(store, Storage) else {}


class Storage(Stub):
    def __setstate__(self, state):
        self.mapping = dict(state.get("_mapping", state)) if isinstance(state, dict) else {}


class Unpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        typename, storage_type, key, location, numel = pid
        return ("storage", key, storage_type, numel)

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *rest: TensorRef(storage[1], storage[2], offset, size, stride)
        if name == "_rebuild_parameter":
            return lambda data, *rest: data
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch_geometric.data.data" and name == "Data":
            return Data
        if module == "torch_geometric.data.storage":
            return Storage
        return type(name, (Stub,), {})


def collect_refs(obj, out):
    """Walk the unpickled object graph and collect every TensorRef (to detect storage sharing)."""
    if isinstance(obj, TensorRef):
        out.append(obj)
    elif isinstance(obj, Data):
        for v in obj.fields.values():
            collect_refs(v, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_refs(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_refs(v, out)
    elif isinstance(obj, Stub) and hasattr(obj, "state"):
        collect_refs(obj.state, out)


def flip_file(path, attrs):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        prefix = names[0].split("/")[0]
        if f"{prefix}/{MARKER}" in names:
            print(f"  skip (already converted): {path.name}")
            return False
        data = Unpickler(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
        if not isinstance(data, list) or not all(isinstance(d, Data) for d in data):
            raise RuntimeError(f"{path}: expected a list of torch_geometric Data objects")

        all_refs = []
        collect_refs(data, all_refs)
        keys_to_flip = {}
        for d in data:
            for attr in attrs:
                ref = d.fields[attr]
                if not isinstance(ref, TensorRef):
                    raise RuntimeError(f"{path}: field {attr} is not a tensor")
                if ref.storage_type not in TYPECODE:
                    raise RuntimeError(f"{path}: field {attr} has unsupported storage {ref.storage_type}")
                keys_to_flip[ref.key] = (attr, ref)
        # safety: a flipped storage must be used by exactly one tensor, fully (no views / sharing)
        usage = {}
        for r in all_refs:
            usage.setdefault(r.key, []).append(r)
        for key, (attr, ref) in keys_to_flip.items():
            users = usage[key]
            raw_numel = len(zf.read(f"{prefix}/data/{key}")) // array.array(TYPECODE[ref.storage_type]).itemsize
            if len(users) != 1 or ref.offset != 0 or ref.numel != raw_numel:
                raise RuntimeError(f"{path}: storage {key} ({attr}) is shared or partially used; refusing to flip")

        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zo:
            for info in zf.infolist():
                payload = zf.read(info.filename)
                key = info.filename[len(prefix) + 6:] if info.filename.startswith(f"{prefix}/data/") else None
                if key in keys_to_flip:
                    arr = array.array(TYPECODE[keys_to_flip[key][1].storage_type])
                    arr.frombytes(payload)
                    payload = array.array(arr.typecode, [-x for x in arr]).tobytes()
                zo.writestr(zipfile.ZipInfo(info.filename, date_time=info.date_time), payload)
            zo.writestr(zipfile.ZipInfo(f"{prefix}/{MARKER}"), b"E_vec, field, polarizability and external_field_v_per_ang follow the physical convention\n")
    path.write_bytes(out.getvalue())
    print(f"  converted: {path.relative_to(ROOT)}  ({len(data)} samples; negated {', '.join(attrs)})")
    return True


def main():
    changed = 0
    for rel, attrs in TARGETS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"  missing: {path}", file=sys.stderr)
            continue
        changed += flip_file(path, attrs)
    print(f"done: {changed} file(s) converted")


if __name__ == "__main__":
    main()
