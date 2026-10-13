# Datasets

The top-level training code uses the **Organic ion-pair dimer** training files
configured in `module.py`. Dataset paths are resolved relative to that module,
not the working directory. Other systems' datasets are included for reference;
the top-level training configuration is not automatically switched between systems.

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

The zero-field 5.5-7.2 Angstrom interval was also evaluated by the original
NaCl validation loader. The merged test file denotes non-training evaluation
data, not necessarily an untouched final holdout.

## NMA

Train, validation, and test data in project-specific PyTorch format.
See [DATA_SOURCE.txt](NMA/DATA_SOURCE.txt) for the FIREANN publication and
the authors' original dataset repository.

## Water molecule

Zero-field single-point, rotation, and electric-field-sweep datasets.

Dataset binaries are copied unchanged from the local project archive.
They contain PyTorch Geometric objects and can be read with
`torch.load(path, weights_only=False, map_location="cpu")` in an environment
with PyTorch and PyTorch Geometric. Only load pickle-based files from trusted sources.
