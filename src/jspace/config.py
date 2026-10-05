"""Pinned model and lens sources.

Every experiment resolves models through :data:`MODELS` so results are tied to
an exact HuggingFace commit. Lenses are either downloaded (``LensSource.hub``)
or fitted locally by ``experiments/00_fit_lens.py`` (``LensSource.local``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
LENS_DIR = REPO_ROOT / "lenses"  # locally fitted lenses (gitignored)


@dataclass(frozen=True)
class LensSource:
    """Where a fitted Jacobian lens lives.

    Exactly one of ``hub`` (``(repo_id, revision, filename)``) or ``local``
    (path relative to :data:`LENS_DIR`) is set.
    """

    hub: tuple[str, str, str] | None = None
    local: str | None = None
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    lenses: dict[str, LensSource]
    default_lens: str
    chat: bool  # post-trained (has a chat template we should use)
    # Workspace band (inclusive layer range), measured by
    # experiments/01_layer_structure.py; None until measured.
    band: tuple[int, int] | None = None

    @property
    def band_layers(self) -> list[int]:
        if self.band is None:
            raise ValueError(f"no workspace band measured for {self.key}")
        return list(range(self.band[0], self.band[1] + 1))


MODELS: dict[str, ModelSpec] = {
    "qwen3.5-4b": ModelSpec(
        key="qwen3.5-4b",
        hf_id="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        lenses={
            # Paper recipe (jlens reference code): 1000 WikiText-103 prompts x 128
            # tokens, target = final block output, skip first 16 positions.
            "np-n1000": LensSource(
                hub=(
                    "neuronpedia/jacobian-lens",
                    "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a",  # branch qwen-n1000
                    "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
                ),
                note="Neuronpedia fit with the jlens reference implementation, n=1000.",
            ),
            # Our own fit (same recipe, fewer prompts) -- an independent check.
            "local": LensSource(local="qwen3.5-4b/lens.pt"),
        },
        default_lens="np-n1000",
        chat=True,
        band=(17, 27),  # results/01_layer_structure/qwen3.5-4b
    ),
    "qwen3.5-4b-base": ModelSpec(
        key="qwen3.5-4b-base",
        hf_id="Qwen/Qwen3.5-4B-Base",
        revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        lenses={"local": LensSource(local="qwen3.5-4b-base/lens.pt")},
        default_lens="local",
        chat=False,
    ),
}
