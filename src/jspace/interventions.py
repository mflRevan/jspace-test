"""Writing to the J-space: residual-stream edits applied by forward hooks.

Each :class:`Edit` acts on the output of a set of decoder blocks at a set of
*absolute* token positions. :class:`Intervene` installs the edits for the
duration of a ``with`` block; generation code keeps ``Intervene.offset`` equal
to the absolute position of the first token in each forward chunk so edits stay
aligned under KV caching. Edits at several layers compose in depth order, which
makes a swap applied at every band layer a "clamped" swap (paper section 3.3).

Implemented edits (paper section 2.5, "Writing"):

* :class:`Steer` -- ``h += alpha * v``.
* :class:`ProjectOut` -- remove the component of ``h`` in ``span{v_i}``.
* :class:`Swap` -- lens-coordinate patch: with ``V = [v_src..., v_tgt...]``
  (unit J-lens vectors), ``c = V^+ h`` and ``h += alpha * V (sigma(c) - c)``
  where ``sigma`` exchanges each source coordinate with its target.
* :class:`TopKAblate` -- per position, project out the ``k`` most active J-lens
  vectors (paper section 3.5.2), optionally protecting some tokens.
* :class:`Clamp` -- hold coordinates along ``V`` at reference values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch

from jspace.model import LensedModel

Positions = Sequence[int] | range | None  # absolute positions; None = every position


def _position_mask(positions: Positions, abs_pos: torch.Tensor) -> torch.Tensor:
    if positions is None:
        return torch.ones_like(abs_pos, dtype=torch.bool)
    if isinstance(positions, range):
        return (abs_pos >= positions.start) & (abs_pos < positions.stop)
    pos = torch.as_tensor(list(positions), device=abs_pos.device)
    return torch.isin(abs_pos, pos)


@dataclass
class Edit:
    layers: Sequence[int]
    positions: Positions = None

    def apply(self, h: torch.Tensor, layer: int, pos: torch.Tensor) -> torch.Tensor:
        """Return edited rows. ``h`` is ``[B, P, d]`` fp32 (the selected rows);
        ``pos`` holds their absolute positions ``[P]``."""
        raise NotImplementedError

    def prepare(self, lm: LensedModel) -> None:  # precompute per-layer tensors
        pass


@dataclass
class Steer(Edit):
    """Add ``alpha * vectors[layer]`` (``[d]``) at the selected positions."""

    vectors: dict[int, torch.Tensor] = field(default_factory=dict)
    alpha: float = 1.0

    def apply(self, h, layer, pos):
        return h + self.alpha * self.vectors[layer].to(h)


@dataclass
class ProjectOut(Edit):
    """Zero the component in ``span(directions[layer])`` (``[k, d]``)."""

    directions: dict[int, torch.Tensor] = field(default_factory=dict)
    _Q: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def prepare(self, lm):
        for l, D in self.directions.items():
            self._Q[l] = torch.linalg.qr(D.float().T).Q.T  # [k, d] orthonormal rows

    def apply(self, h, layer, pos):
        Q = self._Q[layer].to(h)
        return h - (h @ Q.T) @ Q


@dataclass
class Swap(Edit):
    """Exchange lens coordinates of ``src[i]`` and ``tgt[i]`` (token ids)."""

    src: Sequence[int] = ()
    tgt: Sequence[int] = ()
    alpha: float = 1.0
    rtol: float = 1e-4
    _V: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    _Vp: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    _perm: torch.Tensor | None = field(default=None, repr=False)

    def prepare(self, lm):
        assert len(self.src) == len(self.tgt) and len(self.src) > 0
        n = len(self.src)
        ids = list(self.src) + list(self.tgt)
        self._perm = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(lm.device)
        for l in self.layers:
            V = lm.lens_vectors(ids, l, unit=True).T  # [d, 2n]
            self._V[l] = V
            self._Vp[l] = torch.linalg.pinv(V, rtol=self.rtol)  # [2n, d]

    def apply(self, h, layer, pos):
        c = h @ self._Vp[layer].T  # [B, P, 2n]
        return h + self.alpha * (c[..., self._perm] - c) @ self._V[layer].T


@dataclass
class SetSwap(Edit):
    """Exchange the J-space content of two token *sets* (paper Fig. 14 "plan
    swap"): remove each prompt's own coordinates on ``remove`` and install the
    coordinates recorded for ``install`` from another run.

    ``install_coords[layer]`` is ``[T, m]`` -- coordinates on ``install_ids``
    from the donor run, indexed by absolute position (positions past the donor's
    length reuse its last row).
    """

    remove_ids: Sequence[int] = ()
    install_ids: Sequence[int] = ()
    install_coords: dict[int, torch.Tensor] = field(default_factory=dict)
    _Vr: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    _Vi: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    _P: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def prepare(self, lm):
        for l in self.layers:
            V = lm.lens_vectors(list(self.remove_ids) + list(self.install_ids), l, unit=True).T
            self._P[l] = torch.linalg.pinv(V, rtol=1e-4)
            self._Vr[l] = V

    def apply(self, h, layer, pos):
        c = h @ self._P[layer].T  # coords on [remove..., install...]
        nr = len(self.remove_ids)
        new = c.clone()
        new[..., :nr] = 0.0
        donor = self.install_coords[layer].to(h)  # [T_donor, m], by absolute position
        new[..., nr:] = donor[pos.clamp_max(donor.shape[0] - 1)]
        return h + (new - c) @ self._Vr[layer].T


@dataclass
class TopKAblate(Edit):
    """Per position, project out the top-``k`` J-lens directions (by lens logit),
    skipping ``protect`` token ids (paper: tokens in the clean top-10 output)."""

    lm: LensedModel | None = None
    k: int = 10
    protect: Iterable[int] = ()
    _protect: torch.Tensor | None = field(default=None, repr=False)

    def prepare(self, lm):
        self.lm = lm
        self._protect = torch.as_tensor(list(self.protect), dtype=torch.long, device=lm.device)

    def apply(self, h, layer, pos):
        lm = self.lm
        B, P, d = h.shape
        flat = h.reshape(B * P, d)
        logits = lm.lens_logits(flat, layer)
        if self._protect.numel():
            logits[:, self._protect] = float("-inf")
        ids = logits.topk(self.k, dim=-1).indices  # [BP, k]
        w = lm.W_U[ids.flatten()].float() * lm.norm_gain  # [BP*k, d]
        V = (w @ lm.J[layer]).reshape(B * P, self.k, d)
        Q = torch.linalg.qr(V.transpose(1, 2)).Q  # [BP, d, k]
        proj = torch.einsum("nd,ndk->nk", flat, Q)
        return (flat - torch.einsum("nk,ndk->nd", proj, Q)).reshape(B, P, d)


@dataclass
class Clamp(Edit):
    """Hold coordinates on ``token_ids`` at ``coords[layer]`` (``[T, n]``, indexed
    by absolute position), leaving the orthogonal complement untouched."""

    token_ids: Sequence[int] = ()
    coords: dict[int, torch.Tensor] = field(default_factory=dict)
    _V: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    _Vp: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def prepare(self, lm):
        for l in self.layers:
            V = lm.lens_vectors(list(self.token_ids), l, unit=True).T
            self._V[l], self._Vp[l] = V, torch.linalg.pinv(V, rtol=1e-4)

    def apply(self, h, layer, pos):
        c = h @ self._Vp[layer].T
        ref = self.coords[layer].to(h)
        target = ref[pos.clamp_max(ref.shape[0] - 1)]
        return h + (target - c) @ self._V[layer].T


def lens_coords(lm: LensedModel, resid: torch.Tensor, token_ids: Sequence[int], layer: int) -> torch.Tensor:
    """Pseudoinverse lens coordinates ``[T, n]`` of ``resid[layer]`` on unit J-lens vectors."""
    V = lm.lens_vectors(list(token_ids), layer, unit=True).T
    return resid[layer].float() @ torch.linalg.pinv(V, rtol=1e-4).T


class Intervene:
    """Context manager installing ``edits`` as forward hooks on ``lm.layers``."""

    def __init__(self, lm: LensedModel, edits: Iterable[Edit]) -> None:
        self.lm = lm
        self.edits = list(edits)
        self.offset = 0
        self._handles: list = []
        for e in self.edits:
            e.prepare(lm)

    def _hook(self, layer: int) -> Callable:
        edits = [e for e in self.edits if layer in e.layers]

        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            T = h.shape[1]
            abs_pos = torch.arange(self.offset, self.offset + T, device=h.device)
            new = h.float()
            for e in edits:
                sel = _position_mask(e.positions, abs_pos)
                if sel.any():
                    new[:, sel] = e.apply(new[:, sel], layer, abs_pos[sel])
            new = new.to(h.dtype)
            return new if torch.is_tensor(output) else (new, *output[1:])

        return hook

    def __enter__(self) -> Intervene:
        layers = sorted({l for e in self.edits for l in e.layers})
        for l in layers:
            self._handles.append(self.lm.layers[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc) -> None:
        for hd in self._handles:
            hd.remove()
        self._handles = []
