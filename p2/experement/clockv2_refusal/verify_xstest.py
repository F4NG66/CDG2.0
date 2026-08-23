#!/usr/bin/env python3
"""clockv2_refusal/verify_xstest.py — Phase 0.5 verification of the XSTest pull.

READ-ONLY / NO MODEL. Pure text analysis. Nothing is captured, built, or wrapped.
Imports clockv2/common.py READ-ONLY to reconstruct the exact probe fitting sets
(43 / 99 / 40) — no editing of clockv2, no model load (build_heldout is pure text).

Reports, exactly as the PI requested:
  1. Provenance + n per split; binds loader xstest_safe.json to the canonical CSV.
  2. Register stats (FP / OP / length) for BOTH safe and unsafe splits.
  3. Full leak check (exact + Jaccard<=0.4) of BOTH splits vs every probe fitting
     set AND the 100 eval cases.
  4. Form-only classifier (opener/length/FP features) safe-vs-unsafe -> AUC.
     Target ~0.5 == confound B (question-form separability) is fixed.

    /home/ore99/env_llada/bin/python clockv2_refusal/verify_xstest.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "clockv2"))   # read-only import of common
import common  # noqa: E402

RAW = "/scratch/ore99/clockv2_refusal/data/raw"

# ---- register definitions (clockv2 REPORT sec.2) ----
FP_RE = re.compile(r"\b(I|my|me)\b")
def fp(s: str) -> bool: return bool(FP_RE.search(s))
def op(s: str) -> bool: return " ".join(s.split()[:3]).lower() == "how can i"
def nwords(s: str) -> int: return len(s.split())


def reg_stats(name, texts):
    n = len(texts)
    wl = [nwords(t) for t in texts]
    return {
        "name": name, "n": n,
        "FP": sum(fp(t) for t in texts), "OP": sum(op(t) for t in texts),
        "wl_mean": round(statistics.mean(wl), 1), "wl_med": statistics.median(wl),
        "wl_sd": round(statistics.pstdev(wl), 1),
        "char_mean": round(statistics.mean(len(t) for t in texts), 1),
    }


def print_reg(r):
    print(f"  {r['name']:26s} n={r['n']:4d}  FP={r['FP']:3d}/{r['n']:<3d} ({100*r['FP']/r['n']:4.0f}%)  "
          f"OP={r['OP']:3d}/{r['n']:<3d} ({100*r['OP']/r['n']:4.0f}%)  "
          f"words={r['wl_mean']:5.1f}±{r['wl_sd']:<4.1f} (med {r['wl_med']:.0f})  chars={r['char_mean']:.0f}")


# ======================= load XSTest =======================
def load_xstest():
    rows = list(csv.DictReader(open(f"{RAW}/xstest_prompts.csv")))
    safe = [r["prompt"].strip() for r in rows if r["label"] == "safe"]
    unsafe = [r["prompt"].strip() for r in rows if r["label"] == "unsafe"]
    return rows, safe, unsafe


# ======================= leak check =======================
def leak_row(name, pool, ref, thr=0.4):
    sref = set(x.strip() for x in ref)
    exact = sum(1 for t in pool if t.strip() in sref)
    js = [common.max_jaccard_vs(t, ref) for t in pool]
    n_over = sum(1 for j in js if j > thr)
    return name, len(ref), exact, round(max(js), 3) if js else 0.0, n_over


def main():
    print("=" * 78)
    print("PHASE 0.5 — XSTest pull verification (READ-ONLY, no model)")
    print("=" * 78)

    rows, safe, unsafe = load_xstest()

    # ---- 1. provenance ----
    print("\n### 1. Provenance + counts")
    print("  canonical full : Paul/XSTest :: xstest_prompts.csv  (HF hub)")
    print("  loader (safe)  : thu-coai/AISafetyLab_Datasets :: xstest_safe.json")
    print(f"  CSV rows total : {len(rows)}   safe={len(safe)}   unsafe={len(unsafe)}")
    loader = json.load(open(f"{RAW}/xstest_safe.json"))
    loader_prompts = sorted(d["prompt"].strip() for d in loader)
    csv_safe_sorted = sorted(safe)
    match = loader_prompts == csv_safe_sorted
    inter = len(set(loader_prompts) & set(csv_safe_sorted))
    print(f"  loader xstest_safe.json n={len(loader)} ; identical to CSV safe split: {match} "
          f"(intersection {inter}/{len(safe)})")

    # save normalized copy (the pull itself — NOT an eval arm)
    norm = {"source_full": "Paul/XSTest::xstest_prompts.csv",
            "source_loader_safe": "thu-coai/AISafetyLab_Datasets::xstest_safe.json",
            "safe": safe, "unsafe": unsafe}
    outp = os.path.join(HERE, "data", "xstest.json")
    json.dump(norm, open(outp, "w"), indent=1)
    print(f"  normalized copy saved: {outp}")

    # ---- 2. register ----
    print("\n### 2. Register stats (FP / OP / length)")
    rs_safe, rs_unsafe = reg_stats("XSTest SAFE", safe), reg_stats("XSTest UNSAFE", unsafe)
    print_reg(rs_safe); print_reg(rs_unsafe)
    print("  (for reference — the arms these will contrast against:)")
    A = common.eval_behaviors()
    print_reg(reg_stats("eval-100 (DIJA harmful)", A))

    # ---- 3. leak check ----
    print("\n### 3. Leak check  (exact overlap + max Jaccard; near-dup threshold J>0.4)")
    ho = common.build_heldout(j_thresh=0.4, verbose=False)
    harmful43, harmless99 = ho["harmful"], ho["harmless"]
    D_full = [d["behavior"] for d in json.load(open(common.D_NEUTRAL))]
    # tau_bank fit seeds: D_neutral filtered (same filter as harmless99), 40 used
    tau_pool = [b for b in D_full if common.max_jaccard_vs(b, A) <= 0.4][:40]
    unsafe_super = [json.loads(l)["question"] for l in open(common.UNSAFE_MED) if l.strip()]

    refs = [
        ("v_refusal harmful (43)", harmful43),
        ("v_refusal harmless (99)", harmless99),
        ("v_injection_svd fit (43)", harmful43),   # same 43 held-out harmful
        ("tau_bank fit (40)", tau_pool),
        ("D_neutral_clean full (100)", D_full),
        ("unsafe.jsonl superset (50)", unsafe_super),
        ("eval behaviors (100)", A),
    ]
    for split_name, pool in [("SAFE", safe), ("UNSAFE", unsafe)]:
        print(f"\n  -- XSTest {split_name} (n={len(pool)}) vs each fitting set --")
        print(f"     {'fitting set':30s} {'n':>4} {'exact':>6} {'maxJ':>6} {'items J>0.4':>12}")
        worst_exact = worst_j = 0
        for rn, ref in refs:
            _, nref, exact, mj, nover = leak_row(rn, pool, ref)
            worst_exact = max(worst_exact, exact); worst_j = max(worst_j, mj)
            flag = "  <-- LEAK" if (exact > 0 or nover > 0) else ""
            print(f"     {rn:30s} {nref:4d} {exact:6d} {mj:6.3f} {nover:12d}{flag}")
        print(f"     WORST across all sets: exact={worst_exact}  maxJ={worst_j:.3f}  "
              f"-> {'CLEAN' if worst_exact==0 and worst_j<=0.4 else 'CHECK'}")

    # ---- 4. form classifier ----
    print("\n### 4. Form-only classifier  (safe vs unsafe)  — target AUC ~ 0.5")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from scipy.sparse import hstack, csr_matrix

    texts = safe + unsafe
    y = np.array([0] * len(safe) + [1] * len(unsafe))

    # opener features: word 1-2 grams over the FIRST 3 tokens only
    def first3(s): return " ".join(s.split()[:3])
    vec = CountVectorizer(ngram_range=(1, 2), lowercase=True)
    Xop = vec.fit_transform([first3(t) for t in texts])
    # numeric form features: n_words, n_chars, FP flag, OP flag
    Xnum = np.array([[nwords(t), len(t), int(fp(t)), int(op(t))] for t in texts], dtype=float)
    Xnum = (Xnum - Xnum.mean(0)) / (Xnum.std(0) + 1e-9)
    X = hstack([Xop, csr_matrix(Xnum)]).tocsr()
    print(f"  features: {Xop.shape[1]} opener n-grams (first-3-words) + 4 numeric "
          f"(n_words, n_chars, FP, OP) = {X.shape[1]} dims")

    aucs = []
    for seed in (0, 1, 2):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        s = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        aucs.append(s.mean())
        print(f"  seed {seed}: 5-fold CV AUC = {s.mean():.3f}  (folds {np.round(s,3)})")
    print(f"\n  >>> form-only classifier AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
          f"(3 seeds x 5-fold)")
    print(f"  >>> chance = 0.500.  {'PASS — form does NOT separate (confound B fixed)' if abs(np.mean(aucs)-0.5)<=0.10 else 'FAIL — form separates; rebalance needed'}")

    # in-sample reference (upper bound the CV guards against)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    from sklearn.metrics import roc_auc_score
    ins = roc_auc_score(y, clf.decision_function(X))
    print(f"  (in-sample AUC {ins:.3f} — CV number above is the honest one)")


if __name__ == "__main__":
    main()
