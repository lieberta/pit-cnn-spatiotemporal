"""Shared comparison settings; override explicitly in individual run configs."""
seed = 42
split_seed = 42
# None uses <dataset>/experiment_split.json; reuse this file across model seeds.
split_manifest_path = None
# For the paper's source-count protocol on an appropriate dataset, set [3,4,5,6,7].
# Counts outside this list are reserved for test/extrapolation. Steel data uses 18–24.
training_source_counts = None
