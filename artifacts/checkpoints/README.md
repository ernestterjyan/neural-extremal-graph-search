# Release checkpoints

The `mvp-seed-*.pt` files are the best final-stage GNN checkpoints from the
v0.1 curriculum. The `mvp-mlp-seed-*.pt` files are the corresponding fixed-size
MLP controls from v0.2. Every file contains the model state, architecture
configuration, training seed, final curriculum size, selected iteration, and
validation score. The v0.2 files store an explicit model family; the loader
interprets legacy v0.1 files as GNNs through backward-compatible defaults.

Load a checkpoint with `extremal_graph.training.load_model_checkpoint`.
Checkpoint files use PyTorch serialization and should only be loaded from this
trusted repository.
