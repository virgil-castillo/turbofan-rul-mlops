"""Training workflows composing data, features, sequences, and models.

This package owns the application-level train/evaluate use cases (the shared
sequence pipeline, the PyTorch training loop, engine-level splitting, and local
artifact persistence). It composes the lower data/feature/sequence/model layers
rather than belonging to any one of them; ``models`` keeps only estimator and
network definitions.
"""
