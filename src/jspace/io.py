"""Result files with provenance.

Every experiment writes ``results/<name>/results.json`` containing its payload
plus a ``meta`` block (git commit, dirty flag, package versions, model and lens
revisions, seed, wall time) so any number in a report can be traced back.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jspace.config import REPO_ROOT, RESULTS_DIR


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_meta(lm=None, **extra: Any) -> dict:
    meta = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("torch", "transformers", "jlens")},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if lm is not None:
        meta["model"] = {"hf_id": lm.spec.hf_id, "revision": lm.spec.revision}
        meta["lens"] = {"key": lm.lens_key, "source": repr(lm.spec.lenses[lm.lens_key])}
    meta.update(extra)
    return meta


def out_dir(name: str) -> Path:
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _default(o: Any):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def save_results(name: str, payload: dict, meta: dict) -> Path:
    path = out_dir(name) / "results.json"
    path.write_text(json.dumps({"meta": meta, **payload}, indent=1, ensure_ascii=False, default=_default))
    return path
