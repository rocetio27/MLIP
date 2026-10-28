# Datasets

The top-level training code uses the **Organic ion-pair dimer** training files
configured in `module.py`. Dataset paths are resolved relative to that module,
not the working directory. Other systems' datasets are included for reference;
the top-level training configuration is not automatically switched between systems.

## Units and sign conventions

The core model tensors use Hartree atomic units: positions in Bohr, energies in Hartree,
forces in Hartree/Bohr, dipole moments in e·Bohr, polarizabilities in Bohr³
and the external electric field `E_vec` in atomic units (1 a.u. = 51.422 V/Å).

Every dataset follows the physical convention used by the code
(`utils.cal_baseline_energy`, `train_helper.compute_batch_predictions`):

    μ = −∂E/∂ε,   α = −∂²E/∂ε²

The linear induced-dipole response is δμ = α δε. Auxiliary metadata such as
`external_field_v_per_ang` and `d_center_ang` retain the units indicated by their names.

This is **not** the raw convention of the quantum-chemistry inputs: Psi4's
`perturb_dipole` field and FIREANN's `External_field` are the negative of the
physical field, and FIREANN's `POL` is `∂²E/∂F²`, i.e. the negative of the
physical polarizability. The NaCl and ion-pair files were written in the
physical convention when they were generated (`E_vec = −E_psi4`). The water and
NMA files were converted afterwards with `flip_field_sign.py` (repository root),
which negated `E_vec` and `field` in the water files and `E_vec`,
`polarizability` and `external_field_v_per_ang` in the NMA files; each converted
archive carries a `physical_sign_convention` marker record and the script skips
files that already have it. These files are already converted; do not apply an
additional sign reversal when loading them.

## Organic ion-pair dimer

- `dataset_ionic33_train_dimer.pt`: training dimers; the code selects d <= 7.1 Angstrom.
- `dataset_ionic33_train_monomer.pt`: isolated training monomers.
- `dataset_test_group_001.pt`, `dataset_test_group_002.pt`,
  `dataset_test_group_003.pt`: three test groups.
- `dataset_ionic33_4_test_reference_monomer.pt`: test monomer references.

The renamed training files contain the same data as the previously named
`dataset_ionic33_dimer_contact_filtered.pt` and `dataset_ionic33_monomer.pt`.

## NaCl dimer

- `dataset_nacl_train.pt`: 37 zero-field dimers at 1.8-5.4 Angstrom.
- `dataset_nacl_test.pt`: 598 remaining scan configurations, including x- and
  z-directed finite fields. Training structures are excluded from this file.
- `test_reference_isolated_atoms/`: isolated-atom reference data, not part of
  the merged training dataset.

The zero-field 5.5-7.2 Angstrom interval was evaluated during training for
monitoring only; its metrics were not used for parameter updates or checkpoint
selection. Checkpoint selection used only errors on the zero-field training
configurations at 1.8-5.4 Angstrom, logged as "In-range Validation". Despite that
log label, these are the training configurations, not a separate validation set.

## NMA

Train, validation, and test data (10,512 / 1,289 / 1,007 configurations, the
split distributed with FIREANN) in project-specific PyTorch format.
See [DATA_SOURCE.txt](NMA/DATA_SOURCE.txt) for the FIREANN publication and
the authors' original dataset repository. Energies, forces, molecular dipoles
and polarizabilities are FIREANN's labels converted to atomic units; `E_vec`,
`polarizability` and `external_field_v_per_ang` have been converted to the
physical sign convention described above. The NMA files carry **no MBIS
labels**: `charges` and `dipoles` are zero placeholders, and the NMA training
configuration uses `w_q = 0` (no charge-supervision loss).

## Water molecule

The water datasets were generated in this work using DFT calculations performed with Psi4.

Zero-field single-point, rotation, and electric-field-sweep datasets. The
rotation and field-sweep files are evaluation data only; `E_vec` and the scalar
`field` key (z-component of `E_vec`) are in the physical convention.

The dataset files contain PyTorch Geometric objects and can be read with
`torch.load(path, weights_only=False, map_location="cpu")` in an environment
with PyTorch and PyTorch Geometric. Only load pickle-based files from trusted sources.
