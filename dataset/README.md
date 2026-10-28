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

## Organic ion-pair dimer

- `dataset_ionic33_train_dimer.pt`: training dimers; the code selects d <= 7.1 Angstrom.
- `dataset_ionic33_train_monomer.pt`: isolated training monomers.
- `dataset_test_group_001.pt`, `dataset_test_group_002.pt`,
  `dataset_test_group_003.pt`: three test groups.
- `dataset_ionic33_test_reference_monomer.pt`: six isolated-monomer references
  (A and B for each of the three reported test geometries).

## NaCl dimer

- `dataset_nacl_train.pt`: 37 zero-field dimers at 1.8-5.4 Angstrom.
- `dataset_nacl_test.pt`: 598 remaining scan configurations, including x- and
  z-directed finite fields. Training structures are excluded from this file.
- `test_reference_isolated_atoms/`: isolated-atom reference data, not part of
  the merged training dataset.

## NMA

Train, validation, and test data (10,512 / 1,289 / 1,007 configurations, the
split distributed with FIREANN) in project-specific PyTorch format.
See [DATA_SOURCE.txt](NMA/DATA_SOURCE.txt) for the FIREANN publication and
the authors' original dataset repository.

## Water molecule

The water datasets were generated in this work using DFT calculations performed with Psi4.

Zero-field single-point, rotation, and electric-field-sweep datasets. The
rotation and field-sweep files are evaluation data only.
