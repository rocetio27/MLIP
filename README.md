# MLIP

Model implementation and training code for our manuscript "Extrapolating Field Response and Long-Range Interactions with Machine Learning Interatomic Potentials"

## Training

From the repository root, run:

```bash
python train.py
```

The supplied configuration trains on the organic ion-pair dimer and isolated
monomer datasets in `dataset/Organic ion-pair dimer/`. Model settings and dataset
paths are defined in `module.py`; loss weights and training windows are in
`train.py`. Dataset paths are resolved relative to the repository, while
checkpoints are read and written in the working directory.

The code requires PyTorch, PyTorch Geometric (including its `radius_graph`
backend), torch-scatter, e3nn, NumPy, scikit-learn, and tqdm.

## Files

- `train.py`: training entry point.
- `module.py`: model configuration, units, and dataset paths.
- `neural_networks.py`: neural potential and baseline parameters.
- `graph_embedding.py`: node and edge embeddings.
- `blocks.py` and `layers.py`: message-passing and equivariant layers.
- `train_helper.py`: predictions, loss helpers, and evaluation.
- `utils.py`: electrostatics and mathematical utilities.

## Data

See [dataset/README.md](dataset/README.md) for the four dataset collections.
