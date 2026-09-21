#!/usr/bin/env python3
"""p9b_textreader_grouped.py — corrected (leak-free) text-reader baseline for B-vs-A.

Why this script exists
----------------------
`scripts/p9_diagnose.py --task tfidf` reports a TF-IDF "text reader" AUC for the
B-vs-A contrast (harmful-injected vs harmful-clean).  B and A share *byte-identical*
behavior text — only the prompt template differs — so a reader that sees only the
behavior text has nothing legitimate to separate.  Chance is the correct answer.

p9's reader instead uses StratifiedKFold (p9_diagnose.py:207) with no `groups=`.
Each behavior appears exactly twice, once as B (label 1) and once as A (label 0).
When a twin pair is split across folds the model memorises the training copy and
anti-predicts the held-out one, which drives the AUC far *below* chance.

The only change here is the splitter: GroupKFold grouped by the sha256 of the
behavior string actually fed to the vectorizer, so both members of a twin pair
always land in the same fold.  case_id is NOT the right key — it is unique per
(group, case), so grouping on it yields singleton groups and removes nothing.
The vectorizer, the classifier and their hyper-parameters are identical to p9's
reader, so any difference in the reported number is attributable to the splitter
alone.

The record loader is imported from `cdg.probe` directly — the same function
p9_diagnose.py imports at :145 — rather than importing p9_diagnose itself, which
parses argv, chdir()s and creates analysis_output/ at import time.

CPU only: no model, no judge, no GPU, no API calls.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from cdg.data import load_cdg_root          # p9_diagnose.py:144
from cdg.probe import load_records, group_letter   # p9_diagnose.py:145, :270

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

POS_GROUP = "B"   # harmful + injected
NEG_GROUP = "A"   # harmful + clean  (same behavior text)


def text_group_key(text: str) -> str:
    """Group key = sha256 of the exact string handed to the vectorizer.

    This is the true byte-identical equivalence class the twin leak exploits,
    so it binds each B/A pair into one fold by construction — without assuming
    anything about how case ids are formatted.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_rows(out_dir: str, prompt_root: str):
    """Return (texts, labels, groups) for the B-vs-A contrast.

    Rows come from the recorded manifest (load_records); the scored text is the
    *request* text `PromptCase.behavior`, joined in from the prompt corpus by
    (group letter, case_id).  Response text is deliberately not used.
    """
    cases = load_cdg_root(prompt_root)
    behavior_by_key = {}
    dupes = 0
    for c in cases:
        key = (c.group_letter, c.case_id)
        if key in behavior_by_key:
            dupes += 1
        behavior_by_key[key] = c.behavior
    if dupes:
        raise SystemExit(
            f"[p9b] corpus has {dupes} duplicate (group, case_id) key(s) — "
            f"the join would be ambiguous. Aborting.")

    records = load_records(out_dir)
    texts, labels, groups = [], [], []
    missing = 0
    for r in records:
        g = group_letter(r)
        if g not in (POS_GROUP, NEG_GROUP):
            continue
        key = (g, r.get("case_id"))
        if key not in behavior_by_key:
            missing += 1
            continue
        txt = behavior_by_key[key]
        texts.append(txt)
        labels.append(1 if g == POS_GROUP else 0)
        groups.append(text_group_key(txt))

    if missing:
        raise SystemExit(
            f"[p9b] {missing} recorded {POS_GROUP}/{NEG_GROUP} row(s) had no "
            f"matching (group, case_id) in the corpus. Aborting.")
    return texts, np.array(labels), np.array(groups, dtype=object)


def assert_groups_bind_twins(labels, groups):
    """Fail loudly unless every group holds at least one B row and one A row.

    This is the load-bearing check: grouping only defeats the twin leak if the
    B row and its A row really do share the group key.  With sha256(behavior) as
    the key, a clean pass is also a runtime *proof* that the twin behavior texts
    are byte-identical.  Counts only — no ids, no hashes, no text.
    """
    seen = defaultdict(set)
    for lab, grp in zip(labels.tolist(), groups.tolist()):
        seen[grp].add(int(lab))

    pos_only = sum(1 for v in seen.values() if v == {1})
    neg_only = sum(1 for v in seen.values() if v == {0})
    both = sum(1 for v in seen.values() if v == {0, 1})

    print(f"[p9b] groups (sha256 of behavior): {len(seen)}  "
          f"with both {POS_GROUP} and {NEG_GROUP}: {both}  "
          f"{POS_GROUP}-only: {pos_only}  {NEG_GROUP}-only: {neg_only}")

    if pos_only or neg_only:
        raise SystemExit(
            f"[p9b] ASSERTION FAILED: {pos_only + neg_only} of {len(seen)} "
            f"group(s) do not contain both a {POS_GROUP} row and an {NEG_GROUP} "
            f"row. The twin behavior texts are therefore NOT byte-identical, so "
            f"grouping does not remove the leak that p9's StratifiedKFold "
            f"suffers. Fix the premise or the key — do not reinterpret the "
            f"number. Aborting.")
    return len(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/home/ore99/serverFiles/outputs",
                    help="Directory containing manifest.jsonl and the .pt records")
    ap.add_argument("--prompt-root", default="/home/ore99/serverFiles/prompts/cdg_injection",
                    help="4-group prompt corpus root (A/B/C/D subdirectories)")
    ap.add_argument("--out", default="/scratch/ore99/p2_repro/p9b_textreader_grouped.json",
                    help="Where to write the result JSON")
    ap.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits")
    ap.add_argument("--max-features", type=int, default=5000)
    args = ap.parse_args()

    texts, labels, groups = build_rows(args.out_dir, args.prompt_root)

    n = len(texts)
    print(f"[p9b] {POS_GROUP}-vs-{NEG_GROUP} rows: n={n}  "
          f"{POS_GROUP}={int((labels == 1).sum())}  {NEG_GROUP}={int((labels == 0).sum())}")
    if n < 4 or len(set(labels.tolist())) < 2:
        raise SystemExit(f"[p9b] not enough rows or only one class present (n={n}). Aborting.")

    n_groups = assert_groups_bind_twins(labels, groups)
    if n_groups < args.n_splits:
        raise SystemExit(f"[p9b] {n_groups} group(s) < n_splits={args.n_splits}. Aborting.")

    # Identical to p9_diagnose.py:196-199 — only the splitter below differs.
    pipe = make_pipeline(
        TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2),
                        sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=500))

    cv = GroupKFold(n_splits=args.n_splits)
    probs = cross_val_predict(pipe, texts, labels, groups=groups, cv=cv,
                              method="predict_proba")
    pos_idx = int(list(np.unique(labels)).index(1))
    auroc = float(roc_auc_score(labels, probs[:, pos_idx]))

    splitter = f"GroupKFold({args.n_splits}) by sha256(behavior)"
    result = {"auroc_grouped": auroc, "n": n, "splitter": splitter}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[p9b] splitter: {splitter}")
    print(f"[p9b] text-reader AUROC ({POS_GROUP}-vs-{NEG_GROUP}, request text, grouped CV) = {auroc:.4f}")
    print(f"[p9b] wrote {args.out}")


if __name__ == "__main__":
    main()
