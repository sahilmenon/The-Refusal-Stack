from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from refusal_stack.interp.direction import RefusalDirection


def save_refusal_direction(direction: RefusalDirection, artifact_dir: str) -> str:
    slug = direction.model_id.replace("/", "_").replace("-", "_").lower()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Path(artifact_dir) / f"refusal_direction_{slug}_{ts}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": torch.tensor(direction.vector, dtype=torch.float32)},
        str(path),
        metadata={
            "layer_idx": str(direction.layer_idx),
            "norm": str(direction.norm),
            "method": direction.method,
            "model_id": direction.model_id,
            "extraction_split": direction.extraction_split,
            "timestamp": ts,
        }
    )
    return str(path.resolve())


def save_refusal_direction_canonical(direction: RefusalDirection, artifact_dir: str) -> str:
    timestamped_path = save_refusal_direction(direction, artifact_dir)
    canonical = Path(artifact_dir) / "refusal_direction_latest.safetensors"
    canonical.unlink(missing_ok=True)
    try:
        canonical.symlink_to(Path(timestamped_path).name)
    except OSError:
        # Windows (no symlink privilege) raises here; copy the bytes instead so
        # the "latest" pointer still exists rather than killing the run after
        # the expensive extraction/probing.
        import shutil
        shutil.copyfile(timestamped_path, canonical)
    return str(canonical.resolve())


def load_refusal_direction(path: str) -> RefusalDirection:
    tensors = {}
    metadata = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        metadata = f.metadata()
    vec = tensors["direction"].float().numpy()
    return RefusalDirection(
        layer_idx=int(metadata.get("layer_idx", 0)),
        vector=vec,
        norm=float(metadata.get("norm", np.linalg.norm(vec))),
        method=metadata.get("method", "diff_of_means"),
        model_id=metadata.get("model_id", ""),
        extraction_split=metadata.get("extraction_split", "train"),
    )


def save_all_layer_directions(directions: dict[int, RefusalDirection], artifact_dir: str) -> str:
    if not directions:
        return ""
    sample = next(iter(directions.values()))
    slug = sample.model_id.replace("/", "_").replace("-", "_").lower()
    path = Path(artifact_dir) / f"all_layer_directions_{slug}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f"layer_{i:02d}": torch.tensor(d.vector, dtype=torch.float32) for i, d in directions.items()}
    save_file(tensors, str(path))
    return str(path)
