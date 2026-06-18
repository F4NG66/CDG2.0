from __future__ import annotations
"""Probing + SAE semantic analysis over the recorded activations.

Reads the .pt records written by Recorder.save() (plus manifest.jsonl) and
provides:

  * load_records / stack_group        - turn records into (X, y) feature matrices
  * linear_probe                      - logistic-regression detector with CV
                                        AUC / macro-F1 (sklearn, numpy fallback)
  * top_separating_features           - which SAE features separate two groups
  * feature_atlas                     - NxN decoder-direction cosine matrix
                                        (the "Feature Atlas" / orthogonality plot)

The probe answers: "can we DETECT, from the template-region activations, that an
injection is taking effect?"  The default positive/negative split is

    positive = B (harmful + injected, successful)
    negative = C (neutral + injected)

so the probe is forced to separate harmful-fill from benign-fill *within* the
same injection structure - the property steering then removes.
"""
import json
import os
from typing import Optional

import numpy as np
import torch

GROUP_ORDER = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def group_letter(row: dict) -> str:
    v = (row.get("variant") or "")
    return v[:1].upper()


def load_records(out_dir: str, model_name: Optional[str] = None) -> list[dict]:
    """Read manifest.jsonl, attach the full per-case record under '_rec'."""
    man = os.path.join(out_dir, "manifest.jsonl")
    if not os.path.exists(man):
        raise FileNotFoundError(f"manifest not found: {man}")
    rows = []
    with open(man) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if model_name and row.get("model_name") != model_name:
                continue
            path = row.get("path")
            if path and os.path.exists(path):
                try:
                    row["_rec"] = torch.load(path, map_location="cpu",
                                             weights_only=False)
                except Exception as e:  # noqa
                    row["_rec"] = None
                    row["_load_error"] = str(e)
            else:
                row["_rec"] = None
            rows.append(row)
    return rows


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


def _vec_for(rec: dict, scope: str, frac: float, layer: int, space: str):
    key = "hidden" if space == "hidden" else "sae"
    store = rec.get(key, {}).get(scope, {})
    fr = _match_frac(store, frac)
    if fr is None:
        return None
    v = store[fr].get(layer)
    if v is None:
        return None
    v = v.float()
    if torch.isnan(v).any():           # empty region at this frac (e.g. tpl_mask@1.0)
        return None
    return v


def stack_group(records, *, groups, scope, frac, layer, space="hidden"):
    """Stack per-case vectors for the given groups -> (X[n,d], info list)."""
    xs, info = [], []
    for r in records:
        if group_letter(r) not in groups:
            continue
        rec = r.get("_rec")
        if rec is None:
            continue
        v = _vec_for(rec, scope, frac, layer, space)
        if v is None:
            continue
        xs.append(v)
        info.append({"case_id": r.get("case_id"), "variant": r.get("variant"),
                     "judge": r.get("judge")})
    if not xs:
        return None, info
    return torch.stack(xs, 0), info


def build_xy(records, *, scope, frac, layer, space="hidden",
             pos_groups=("B",), neg_groups=("C",), success_only=False):
    """Assemble a labelled matrix for a binary detection probe."""
    Xp, ip = stack_group(records, groups=pos_groups, scope=scope, frac=frac,
                         layer=layer, space=space)
    Xn, ineg = stack_group(records, groups=neg_groups, scope=scope, frac=frac,
                          layer=layer, space=space)
    if Xp is None or Xn is None:
        return None, None, (ip, ineg)
    if success_only:
        keep = [k for k, info in enumerate(ip)
                if (info.get("judge") or {}).get("success") != 0]
        if keep:
            Xp = Xp[keep]
    X = torch.cat([Xp, Xn], 0).numpy()
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]).astype(int)
    return X, y, (ip, ineg)


# ---------------------------------------------------------------------------
# linear probe
# ---------------------------------------------------------------------------
def linear_probe(X, y, *, cv: int = 5, C: float = 1.0, seed: int = 0) -> dict:
    """Cross-validated logistic regression. Returns AUC + macro-F1 (+std)."""
    n = len(y)
    if n < 4 or len(set(y.tolist())) < 2:
        return {"auc": float("nan"), "macro_f1": float("nan"), "n": int(n),
                "note": "insufficient data / single class"}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score, f1_score

        k = min(cv, np.bincount(y).min())
        if k < 2:
            return {"auc": float("nan"), "macro_f1": float("nan"), "n": int(n),
                    "note": "a class has <2 samples"}
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        aucs, f1s = [], []
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=C, max_iter=2000))
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            try:
                aucs.append(roc_auc_score(y[te], prob))
            except ValueError:
                pass
            f1s.append(f1_score(y[te], pred, average="macro"))
        return {"auc": float(np.mean(aucs)) if aucs else float("nan"),
                "auc_std": float(np.std(aucs)) if aucs else float("nan"),
                "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                "n": int(n), "n_splits": int(k), "backend": "sklearn"}
    except ImportError:
        return _numpy_probe(X, y, seed=seed)


def _numpy_probe(X, y, *, seed=0, iters=300, lr=0.1) -> dict:
    """Tiny GD logistic regression fallback (no sklearn)."""
    rng = np.random.RandomState(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    idx = rng.permutation(len(y))
    cut = max(1, int(0.7 * len(y)))
    tr, te = idx[:cut], idx[cut:]
    if len(te) == 0 or len(set(y[te].tolist())) < 2:
        te = tr
    w = np.zeros(Xs.shape[1]); b = 0.0
    for _ in range(iters):
        z = Xs[tr] @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y[tr]
        w -= lr * (Xs[tr].T @ g / len(tr) + 1e-3 * w)
        b -= lr * g.mean()
    prob = 1 / (1 + np.exp(-(Xs[te] @ w + b)))
    pred = (prob >= 0.5).astype(int)
    auc = _auc(y[te], prob)
    f1 = _macro_f1(y[te], pred)
    return {"auc": auc, "macro_f1": f1, "n": int(len(y)), "backend": "numpy"}


def _auc(y, s) -> float:
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def _macro_f1(y, pred) -> float:
    f1s = []
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def probe_sweep(records, *, scopes, fracs, layers, space="hidden",
                pos_groups=("B",), neg_groups=("C",), success_only=False) -> list[dict]:
    """Grid over (scope, frac, layer) -> probe metrics. Handy for the paper's
    'where/when is injection most linearly decodable' figure."""
    rows = []
    for scope in scopes:
        for frac in fracs:
            for layer in layers:
                X, y, _ = build_xy(records, scope=scope, frac=frac, layer=layer,
                                   space=space, pos_groups=pos_groups,
                                   neg_groups=neg_groups, success_only=success_only)
                if X is None:
                    rows.append({"scope": scope, "frac": frac, "layer": layer,
                                 "auc": float("nan"), "macro_f1": float("nan"),
                                 "n": 0})
                    continue
                m = linear_probe(X, y)
                m.update({"scope": scope, "frac": frac, "layer": layer})
                rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# SAE semantic analysis
# ---------------------------------------------------------------------------
def top_separating_features(records, *, scope, frac, layer,
                            pos_groups=("B",), neg_groups=("C",), k: int = 20) -> dict:
    """Rank SAE features by mean-activation gap between pos and neg groups.

    Returns indices + the signed gap; positive gap = feature fires more under
    the harmful-injection group, i.e. a candidate 'injection feature'.
    """
    Xp, _ = stack_group(records, groups=pos_groups, scope=scope, frac=frac,
                        layer=layer, space="sae")
    Xn, _ = stack_group(records, groups=neg_groups, scope=scope, frac=frac,
                        layer=layer, space="sae")
    if Xp is None or Xn is None:
        return {"note": "insufficient data", "features": []}
    mp = Xp.mean(0).numpy(); mn = Xn.mean(0).numpy()
    gap = mp - mn
    order = np.argsort(-np.abs(gap))[:k]
    feats = [{"feature": int(i), "gap": float(gap[i]),
              "pos_mean": float(mp[i]), "neg_mean": float(mn[i])} for i in order]
    return {"scope": scope, "frac": frac, "layer": layer,
            "n_pos": int(Xp.shape[0]), "n_neg": int(Xn.shape[0]),
            "features": feats}


def top_diff_features(
    records,
    *,
    scope: str = "tpl_mask",
    fracs: Optional[tuple] = None,
    layers: Optional[tuple] = None,
    pairs: tuple = (("B", "C"), ("A", "B"), ("C", "D")),
    k: int = 30,
    stat: str = "mean_gap",
) -> list[dict]:
    """Sweep over (layer × frac × group_pair) and return the top-k differentially
    activated SAE features for each combination.

    Args:
        pairs: group-letter pairs to compare, e.g. ("B","C") = harmful-inj vs neutral-inj.
               The gap is signed: positive = first group fires more.
        stat:  "mean_gap"  - difference of per-group mean activations (default)
               "t_stat"    - Welch t-statistic (more robust for unequal sizes)

    Returns:
        list of dicts, one per (layer, frac, pair) entry that has data.
        Each dict has keys: layer, frac, pair, n_pos, n_neg, features (list of k dicts).
    """
    # discover available fracs/layers from first matching record
    if fracs is None or layers is None:
        for r in records:
            rec = r.get("_rec")
            if rec is None:
                continue
            store = rec.get("sae", {}).get(scope, {})
            if store:
                if fracs is None:
                    fracs = tuple(sorted(float(f) for f in store))
                if layers is None:
                    any_frac = next(iter(store.values()))
                    layers = tuple(sorted(any_frac.keys()))
                break
    if not fracs or not layers:
        return []

    rows = []
    for layer in layers:
        for frac in fracs:
            for pos_grp, neg_grp in pairs:
                Xp, _ = stack_group(records, groups=(pos_grp,), scope=scope,
                                    frac=frac, layer=layer, space="sae")
                Xn, _ = stack_group(records, groups=(neg_grp,), scope=scope,
                                    frac=frac, layer=layer, space="sae")
                if Xp is None or Xn is None:
                    continue
                mp = Xp.float().mean(0).numpy()
                mn = Xn.float().mean(0).numpy()

                if stat == "t_stat" and Xp.shape[0] > 1 and Xn.shape[0] > 1:
                    sp = Xp.float().numpy()
                    sn = Xn.float().numpy()
                    var_p = sp.var(0) / sp.shape[0]
                    var_n = sn.var(0) / sn.shape[0]
                    denom = np.sqrt(var_p + var_n + 1e-12)
                    score = (mp - mn) / denom
                else:
                    score = mp - mn

                order = np.argsort(-np.abs(score))[:k]
                feats = [
                    {"feature": int(i),
                     "score": float(score[i]),
                     "pos_mean": float(mp[i]),
                     "neg_mean": float(mn[i])}
                    for i in order
                ]
                rows.append({
                    "layer": int(layer), "frac": float(frac),
                    "pair": f"{pos_grp}vs{neg_grp}",
                    "pos_group": pos_grp, "neg_group": neg_grp,
                    "stat": stat,
                    "n_pos": int(Xp.shape[0]), "n_neg": int(Xn.shape[0]),
                    "features": feats,
                })
    return rows


def diff_feature_matrix(records, *, scope: str = "tpl_mask",
                        frac: float = 0.10, layer: int = 16,
                        pairs: tuple = (("B", "C"), ("A", "B"), ("C", "D")),
                        k: int = 20) -> dict:
    """Convenience: for a single (layer, frac), return a matrix view showing
    the top-k features per pair side-by-side for quick comparison.

    Returns dict with 'pairs' key and per-pair feature lists, plus a 'union'
    list of feature indices that appear in ANY pair's top-k (useful for atlas).
    """
    rows = top_diff_features(records, scope=scope, fracs=(frac,), layers=(layer,),
                             pairs=pairs, k=k)
    union = set()
    result = {}
    for r in rows:
        feat_ids = [f["feature"] for f in r["features"]]
        result[r["pair"]] = r
        union.update(feat_ids)
    return {"layer": layer, "frac": frac, "scope": scope,
            "pairs": result, "union_features": sorted(union)}


def feature_atlas(sae, feature_ids) -> np.ndarray:
    """Cosine-similarity matrix between SAE decoder directions of `feature_ids`.

    Near-orthogonal off-diagonals (|cos| small) support the paper's claim that
    the injection-relevant features are distinct, monosemantic directions.
    Requires a loaded TopKSAE (so call on the analysis box, not from records).
    """
    dirs = torch.stack([sae.feature_direction(int(f)) for f in feature_ids], 0)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    M = (dirs @ dirs.t()).cpu().numpy()
    return M


def save_analysis(rows_or_obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows_or_obj, f, indent=2, default=float)
    return path
