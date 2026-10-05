"""Load a model together with its Jacobian lens.

:class:`LensedModel` is the one object experiments pass around. It wraps the
HuggingFace model in the ``jlens`` reference adapter (so fitting and readout
match the paper's code exactly) and adds what interventions need: the per-layer
``J_l`` on the GPU, the effective unembedding, and J-lens vectors.

Conventions
-----------
* Layer ``l`` means the residual stream at the *output* of decoder block ``l``
  (``0 <= l < n_layers``). Block ``n_layers - 1`` is the lens target, so lenses
  exist for ``l < n_layers - 1``.
* The J-lens vector for token ``t`` at layer ``l`` is row ``t`` of
  ``W_eff @ J_l`` with ``W_eff = W_U diag(g)``, where ``g`` is the gain of the
  final RMSNorm. Up to the data-dependent RMS factor, ``<v_t, h_l>`` is the
  lens logit of ``t`` (paper section 2.5, "Reading").
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import jlens
import torch
import transformers
from huggingface_hub import hf_hub_download
from jlens.hooks import ActivationRecorder

from jspace.config import LENS_DIR, MODELS, ModelSpec

logger = logging.getLogger(__name__)


def load_lens(spec: ModelSpec, lens_key: str | None = None) -> jlens.JacobianLens:
    source = spec.lenses[lens_key or spec.default_lens]
    if source.hub is not None:
        repo_id, revision, filename = source.hub
        path = hf_hub_download(repo_id, filename, revision=revision)
    else:
        path = LENS_DIR / source.local
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; fit it with experiments/00_fit_lens.py")
    return jlens.JacobianLens.load(str(path))


@dataclass
class Forward:
    """Residual stream for one prompt at every layer, plus final logits."""

    ids: torch.Tensor  # [T]
    resid: torch.Tensor  # [n_layers, T, d_model], model dtype, on device
    logits: torch.Tensor  # [T, vocab] fp32 on device


class LensedModel:
    def __init__(
        self,
        model_key: str = "qwen3.5-4b",
        lens_key: str | None = None,
        *,
        load_lens_matrices: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.spec = MODELS[model_key]
        self.lens_key = lens_key or self.spec.default_lens
        self.hf = transformers.AutoModelForCausalLM.from_pretrained(
            self.spec.hf_id, revision=self.spec.revision, dtype=dtype
        ).to(device)
        self.tok = transformers.AutoTokenizer.from_pretrained(
            self.spec.hf_id, revision=self.spec.revision
        )
        self.lm = jlens.from_hf(self.hf, self.tok)  # sets requires_grad=False, eval()
        self.layers = self.lm.layers
        self.n_layers: int = self.lm.n_layers
        self.d_model: int = self.lm.d_model
        self.device = torch.device(device)

        # Effective per-dimension gain of the final RMSNorm, measured on an
        # all-ones input (RMS 1), so zero-centred variants (Qwen3.5 uses
        # ``x * (1 + w)``) and standard ones are handled alike.
        with torch.no_grad():
            ones = torch.ones(1, self.d_model, device=device, dtype=torch.float32)
            self.norm_gain = self.lm._final_norm.float()(ones)[0].clone()
            self.lm._final_norm.to(dtype)
        self.W_U = self.lm._lm_head.weight  # [vocab, d], model dtype
        self.vocab_size = self.W_U.shape[0]

        self.lens: jlens.JacobianLens | None = None
        self.J: dict[int, torch.Tensor] = {}
        if load_lens_matrices:
            self.lens = load_lens(self.spec, self.lens_key)
            self.J = {l: J.to(device) for l, J in self.lens.jacobians.items()}
        self.lens_layers = sorted(self.J)
        self._lens_vec_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    # ------------------------------------------------------------------ text
    def encode(self, text: str) -> torch.Tensor:
        """Tokenize to a 1-D id tensor on device (no truncation)."""
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
        return ids.to(self.device)

    def decode(self, ids: Sequence[int] | torch.Tensor) -> str:
        return self.tok.decode(list(map(int, ids)))

    def token_strs(self, ids: Sequence[int] | torch.Tensor) -> list[str]:
        return [self.tok.decode([int(i)]) for i in ids]

    def chat(
        self,
        messages: str | list[dict],
        *,
        system: str | None = None,
        prefill: str | None = None,
        thinking: bool = False,
    ) -> str:
        """Format a conversation with the chat template.

        ``messages`` is a user string or a list of ``{"role", "content"}``. If
        ``prefill`` is given (``""`` allowed) the assistant turn is opened and
        left unterminated so the model continues it. Thinking is disabled by
        default so the assistant turn begins with the answer itself.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        if not self.spec.chat:
            return _plain_chat(msgs, prefill)
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        return text + (prefill or "")

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def run(self, ids: torch.Tensor | str) -> Forward:
        """One forward pass recording the residual at every block output."""
        if isinstance(ids, str):
            ids = self.encode(ids)
        with ActivationRecorder(self.layers, at=range(self.n_layers)) as rec:
            self.lm.forward(ids[None])
            resid = torch.stack([rec.activations[l][0] for l in range(self.n_layers)])
        logits = self.lm.unembed(resid[-1]).float()
        return Forward(ids=ids, resid=resid, logits=logits)

    # ----------------------------------------------------------------- lens
    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """``J_l h`` for ``h`` of shape ``[..., d]`` (identity at the target layer)."""
        if layer == self.n_layers - 1:
            return h.float()
        return h.float() @ self.J[layer].T

    def lens_logits(self, h: torch.Tensor, layer: int, *, method: str = "jacobian") -> torch.Tensor:
        """Lens logits ``W_U norm(J_l h)``; ``method="logit"`` uses ``J = I``."""
        x = self.transport(h, layer) if method == "jacobian" else h.float()
        return self.lm.unembed(x).float()

    def lens_vectors(self, token_ids: Sequence[int], layer: int, *, unit: bool = False) -> torch.Tensor:
        """J-lens vectors ``[n, d]`` (fp32): rows of ``W_eff J_l`` for ``token_ids``."""
        key = (layer, tuple(int(t) for t in token_ids))
        vecs = self._lens_vec_cache.get(key)
        if vecs is None:
            w = self.W_U[list(key[1])].float() * self.norm_gain  # [n, d]
            vecs = w if layer == self.n_layers - 1 else w @ self.J[layer]
            if len(self._lens_vec_cache) < 4096:
                self._lens_vec_cache[key] = vecs
        return vecs / vecs.norm(dim=-1, keepdim=True) if unit else vecs

    def lens_dictionary(self, layer: int, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """All J-lens vectors at ``layer`` as unit rows ``[vocab, d]`` (large; ~1.3 GB)."""
        out = torch.empty(self.vocab_size, self.d_model, dtype=dtype, device=self.device)
        for s in range(0, self.vocab_size, 32768):
            w = self.W_U[s : s + 32768].float() * self.norm_gain
            v = w @ self.J[layer] if layer != self.n_layers - 1 else w
            out[s : s + 32768] = (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(dtype)
        return out

    def mean_resid_norm(self, layer: int, sample: Forward) -> float:
        """Mean residual norm at ``layer`` over a sample forward (skipping position 0)."""
        return sample.resid[layer, 1:].float().norm(dim=-1).mean().item()


def _plain_chat(msgs: list[dict], prefill: str | None) -> str:
    """Plain-text dialogue format for base models (paper section 6 uses the same
    transcript text for base and post-trained models; we use a neutral
    Human/Assistant rendering)."""
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    parts = [f"{names[m['role']]}: {m['content']}" for m in msgs]
    text = "\n\n".join(parts) + "\n\nAssistant:"
    return text + ((" " + prefill.lstrip()) if prefill else "")
