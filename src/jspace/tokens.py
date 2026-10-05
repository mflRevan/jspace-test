"""Mapping words to vocabulary tokens.

The J-lens has one vector per vocabulary token, so a "concept" such as *spider*
is tracked through its single-token surface forms (``" spider"``, ``"spider"``,
``" Spider"``...). Following the paper, a concept's rank at a cell is the best
(minimum) rank over its forms. Words with no single-token form fall back to
their first token only when ``allow_prefix`` is set (paper section 9.1 notes the
lens then sees fragments such as ``black`` for *blackmail*).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

import torch


def surface_forms(word: str) -> list[str]:
    w = word.strip()
    cands = [w, w.lower(), w.capitalize(), w.upper() if len(w) <= 4 else None]
    out: list[str] = []
    for c in cands:
        if c is None:
            continue
        for s in (" " + c, c):
            if s not in out:
                out.append(s)
    return out


def single_token_ids(tok, word: str, *, allow_prefix: bool = False) -> list[int]:
    """Ids of every single-token surface form of ``word`` (may be empty)."""
    ids: list[int] = []
    for s in surface_forms(word):
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    if not ids and allow_prefix:
        for s in (" " + word.strip(), word.strip()):
            enc = tok.encode(s, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
    return ids


def primary_token_id(tok, word: str) -> int | None:
    """The single token for ``word`` as it would appear mid-sentence
    (leading space preferred), or ``None`` if it is multi-token."""
    for s in (" " + word.strip(), word.strip()):
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            return enc[0]
    return None


class WordSets:
    """A fixed list of concepts, each a set of token ids, for batched ranking."""

    def __init__(self, tok, words: Iterable[str], *, allow_prefix: bool = False) -> None:
        self.words: list[str] = []
        self.ids: list[list[int]] = []
        self.missing: list[str] = []
        for w in words:
            ids = single_token_ids(tok, w, allow_prefix=allow_prefix)
            if ids:
                self.words.append(w)
                self.ids.append(ids)
            else:
                self.missing.append(w)

    def __len__(self) -> int:
        return len(self.words)

    def flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(token_ids, owner)``: all ids concatenated and the word index of each."""
        ids = [t for group in self.ids for t in group]
        owner = [i for i, group in enumerate(self.ids) for _ in group]
        return torch.tensor(ids), torch.tensor(owner)


_WORDLIKE = re.compile(r"[^\W_]", re.UNICODE)  # a letter or digit (not "_")


@lru_cache(maxsize=4)
def _wordlike_mask_cached(tok_id: int, vocab_size: int, tok) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(min(vocab_size, len(tok))):
        s = tok.decode([i]).strip()
        if len(s) >= 2 and _WORDLIKE.search(s) and not s.startswith("<|"):
            mask[i] = True
        elif len(s) == 1 and (s.isdigit() or unicodedata.category(s).startswith("Lo")):
            mask[i] = True  # digits and CJK single characters
    return mask


def wordlike_mask(tok, vocab_size: int) -> torch.Tensor:
    """Vocab mask of tokens that read as words (>=2 chars with a word character,
    or a digit / CJK character). Used only to filter *displayed* top-k lists;
    ranks are always over the full vocabulary, as in the reference code."""
    return _wordlike_mask_cached(id(tok), vocab_size, tok)


def paired_forms(tok, a: str, b: str, *, max_pairs: int = 4) -> tuple[list[int], list[int]]:
    """Aligned single-token surface forms of ``a`` and ``b`` for lens swaps:
    the i-th id of each list is the same casing/spacing variant (e.g.
    ``" spider"``/``" ant"``, ``"Spider"``/``"Ant"``). Forms that are single
    tokens for only one of the words are dropped."""
    src, tgt = [], []
    for fa, fb in zip(surface_forms(a), surface_forms(b), strict=False):
        ea = tok.encode(fa, add_special_tokens=False)
        eb = tok.encode(fb, add_special_tokens=False)
        if len(ea) == 1 and len(eb) == 1 and ea[0] not in src and eb[0] not in tgt and ea[0] != eb[0]:
            src.append(ea[0])
            tgt.append(eb[0])
        if len(src) == max_pairs:
            break
    return src, tgt
