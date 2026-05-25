"""Helpers for loading dataset version manifests."""

import json
from pathlib import Path


def infer_manifest_path(dataset_path: Path) -> Path | None:
    """Infer a version manifest path for a known dataset file."""
    name = dataset_path.name
    if name == "mesa_v2.json":
        return dataset_path.with_name("version_v2.json")
    return None


def load_dataset_manifest(dataset_path: Path) -> dict | None:
    """Load a dataset manifest if one exists for the dataset path."""
    manifest_path = infer_manifest_path(dataset_path)
    if manifest_path is None or not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)
