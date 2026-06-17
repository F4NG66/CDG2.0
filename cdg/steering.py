from __future__ import annotations
"""Build residual-space steering vectors from recorded activations.

The injection direction is identified by a *difference-in-differences* logic:

    (B - A) - (C - D)        # in any region shared by all four groups
                             #   B harmful+template, A harmful+clean,
                             #   C neutral+template, D neutral+clean

but the vector we actually *apply* lives at the INJECTED TEMPLATE positions,
which only exist for the two injected groups (B, C).  So the operational
default is the within-template contrast

    v = mean_hidden(B, tpl_mask) - mean_hidden(C, tpl_mask)

i.e. "what is different about the blanks when the scaffold is being filled with
HARMFUL content vs. BENIGN content".  Subtracting this (alpha < 0) at the
template-mask positions should suppress harmful compliance while leaving the
benign fill ability (C, D) - and thus general capability - intact.  We steer in
the residual stream because that is where HookManager injects.

Everything is computed from the .pt records written by Recorder.save(); no model
is required to *build* the vectors, only to *apply* them (done in dlm_runner).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .probe import load_records, group_letter, stack_group  # reuse loaders


@dataclass
class SteeringVectors:
    """Per-layer steering directions plus the metadata needed to apply them."""
    scope: str                      # region/scope the vectors were built on
    space: str                      # "hidden" (residual) | "sae"
    frac: float
    pos_groups: tuple
    neg_groups: tuple
    layers: tuple
    vectors: dict = field(default_factory=dict)   # layer:int -> 1D tensor[d]
    norms: dict = field(default_factory=dict)      # layer:int -> float (pre-norm)
    meta: dict = field(default_factory=dict)

    # -- io -----------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.__dict__, path)
        side = os.path.splitext(path)[0] + ".json"
        with open(side, "w") as f:
            json.dump({"scope": self.scope, "space": self.space, "frac": self.frac,
                       "pos_groups": list(self.pos_groups),
                       "neg_groups": list(self.neg_groups),
                       "layers": list(self.layers), "norms": self.norms,
                       "meta": self.meta}, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "SteeringVectors":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**d)

    def as_runner_dict(self, normalize: bool = True) -> dict:
        """Return {layer -> unit (or raw) vector} for DLMRunner.set_steering."""
        out = {}
        for layer, v in self.vectors.items():
            v = v.float()
            if normalize:
                v = v / (v.norm() + 1e-8)
            out[int(layer)] = v
        return out


def _mean_over_cases(records, groups, scope, frac, layer, space):
    """Mean activation vector for `groups` at (scope, frac, layer)."""
    X, _ = stack_group(records, groups=groups, scope=scope, frac=frac,
                       layer=layer, space=space)
    if X is None or X.shape[0] == 0:
        return None
    return X.mean(0)


def build_steering_vectors(
    out_dir: str,
    model_name: str,
    *,
    scope: str = "tpl_mask",
    space: str = "hidden",
    frac: float = 0.10,
    pos_groups: tuple = ("B",),
    neg_groups: tuple = ("C",),
    layers: Optional[tuple] = None,
    success_only: bool = True,
    records=None,
) -> SteeringVectors:
    """Mean-difference steering direction per layer.

    pos_groups - groups whose mean we move AWAY from (default B: harmful+injected)
    neg_groups - reference groups (default C: neutral+injected)
    success_only - if True, restrict pos_groups to judged-successful injections
                   (judge.success == 1), so the vector targets the *effective*
                   attack direction rather than attempted-but-refused cases.
    """
    if records is None:
        records = load_records(out_dir, model_name)
    if not records:
        raise RuntimeError(f"no records under {out_dir} for model={model_name}")

    # discover layers present in the first record's scope
    if layers is None:
        layers = _discover_layers(records, scope, frac, space)
        if not layers:
            raise RuntimeError(f"no layers found for scope={scope} frac={frac}")

    pos_records = records
    if success_only:
        kept = [r for r in records
                if not (group_letter(r) in pos_groups
                        and (r.get("judge") or {}).get("success") == 0)]
        # only filters the positive side; negatives are untouched below
        pos_records = kept

    vectors, norms = {}, {}
    for layer in layers:
        mu_pos = _mean_over_cases(pos_records, pos_groups, scope, frac, layer, space)
        mu_neg = _mean_over_cases(records, neg_groups, scope, frac, layer, space)
        if mu_pos is None or mu_neg is None:
            continue
        v = (mu_pos - mu_neg).float()
        norms[int(layer)] = float(v.norm())
        vectors[int(layer)] = v

    if not vectors:
        raise RuntimeError("could not build any steering vector "
                           "(check that pos/neg groups have data at this scope/frac)")

    return SteeringVectors(
        scope=scope, space=space, frac=frac,
        pos_groups=tuple(pos_groups), neg_groups=tuple(neg_groups),
        layers=tuple(sorted(vectors)), vectors=vectors, norms=norms,
        meta={"model_name": model_name, "success_only": success_only,
              "n_records": len(records)},
    )


def build_did_vectors(
    out_dir: str,
    model_name: str,
    *,
    space: str = "hidden",
    frac: float = 1.0,
    scope: str = "out_unmask",
    layers: Optional[tuple] = None,
    records=None,
) -> SteeringVectors:
    """Difference-in-differences vector  (B - A) - (C - D)  on a *shared* region.

    Use scope="out_unmask" (the answer region exists for all four groups) for the
    fully-identified injection-compliance direction.  This is a good ablation
    against the simpler within-template B-C vector.
    """
    if records is None:
        records = load_records(out_dir, model_name)
    if layers is None:
        layers = _discover_layers(records, scope, frac, space)

    vectors, norms = {}, {}
    for layer in layers:
        muB = _mean_over_cases(records, ("B",), scope, frac, layer, space)
        muA = _mean_over_cases(records, ("A",), scope, frac, layer, space)
        muC = _mean_over_cases(records, ("C",), scope, frac, layer, space)
        muD = _mean_over_cases(records, ("D",), scope, frac, layer, space)
        if any(m is None for m in (muA, muB, muC, muD)):
            continue
        v = ((muB - muA) - (muC - muD)).float()
        norms[int(layer)] = float(v.norm())
        vectors[int(layer)] = v
    if not vectors:
        raise RuntimeError("DiD vector build failed; need all of A/B/C/D at this scope")
    return SteeringVectors(
        scope=scope, space=space, frac=frac,
        pos_groups=("B", "A"), neg_groups=("C", "D"),
        layers=tuple(sorted(vectors)), vectors=vectors, norms=norms,
        meta={"model_name": model_name, "kind": "did", "n_records": len(records)},
    )


def _discover_layers(records, scope, frac, space) -> tuple:
    key = "hidden" if space == "hidden" else "sae"
    for r in records:
        rec = r.get("_rec")
        if rec is None:
            continue
        store = rec.get(key, {}).get(scope, {})
        # frac keys may be float; match tolerant
        fr = _match_frac(store, frac)
        if fr is not None and store.get(fr):
            return tuple(sorted(store[fr].keys()))
    return tuple()


def _match_frac(store: dict, frac: float):
    if frac in store:
        return frac
    for k in store:
        try:
            if abs(float(k) - float(frac)) < 1e-6:
                return k
        except (TypeError, ValueError):
            continue
    return None
