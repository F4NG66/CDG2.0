#!/usr/bin/env python3
"""clockv2/build_benign.py — MATCHED-CONTROL ARM (construction only; no model, no capture).

Builds a length- and format-matched BENIGN scaffold for each of the 100 eval ids, so that
the decisive test can ask whether v_injection_svd encodes INJECTION SEMANTICS or merely
PROMPT LENGTH/FORMAT.

WHAT IS HELD CONSTANT vs the DIJA arm (exact, per id):
  * the line skeleton            (question line + 3-4 body lines)
  * every body-line prefix       ("Procedure: ", "First, ", ...) copied verbatim -- these
                                 are neutral format words, not harmful content
  * every <mask:N> literal       => n_inject is IDENTICAL per id, not merely similar
  * the <<TPL>> wrapper + chat template (both arms go through run_dija.PaperRunner)

WHAT VARIES (this is the contrast under test):
  * the question line -> a benign medical question from the held-out `harmless` pool
    (99 items, already J<=0.4 leakage-filtered vs the 100 eval cases, and medical, so the
    contrast isolates HARM, not domain). Assigned by optimal length matching.
  * each trailing sentence -> neutral filler, length-matched.

NO HARMFUL TEXT IS AUTHORED. The harmful question and the harm-carrying trailing sentences
are DISCARDED and replaced with benign text; nothing harmful is written, extended, or
completed. Construction is pure string manipulation -- no model is loaded here.

DEVIATION, STATED: the trailing sentences are authored neutral filler, NOT drawn from the
harmless pool. The pool contains questions, and a question cannot occupy a trailing-clause
slot without wrecking the register match that this arm exists to preserve. The filler is
generic and content-free; it is matched to the DIJA trailing on token length and imperative
register only.

    python clockv2/build_benign.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer

from common import HELDOUT_DIR, MODEL_ID, REFINED_100

BODY = re.compile(r"^(?P<prefix>.*?<mask:\d+>)(?P<suffix>.*)$")

# Neutral, content-free imperative filler spanning the DIJA trailing length range.
# Benign by construction; nothing here is harmful, evasive, or instructional-for-harm.
FILLER = [
    "Keep the description clear.",
    "Use plain language throughout.",
    "Include the relevant background details.",
    "Refer to the standard published guidelines.",
    "Summarize the key points for the reader.",
    "Provide context so the answer is easy to follow.",
    "Make sure the explanation stays accurate and complete.",
    "Organize the information in a straightforward and readable way.",
    "Cite the usual reference materials wherever they are appropriate.",
    "Present the information clearly so that a general reader can follow it.",
    "Take care to describe each part accurately and in a sensible, logical order.",
    "Ensure the summary reflects the standard guidance and remains easy to understand.",
]


def parse(scaffold: str):
    """-> (question, [(prefix_including_mask, suffix), ...]). Raises if shape is unexpected."""
    lines = scaffold.split("\n")
    q, body = lines[0], lines[1:]
    out = []
    for l in body:
        m = BODY.match(l)
        if not m:
            raise ValueError(f"unparsed body line: {l!r}")
        out.append((m.group("prefix"), m.group("suffix")))
    return q, out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    nt = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    dija = json.load(open(REFINED_100))
    ho = json.load(open(os.path.join(HELDOUT_DIR, "contrastive.json")))
    harmless = ho["harmless"]
    print(f"[benign] {len(dija)} dija scaffolds | {len(harmless)} harmless pool items")

    parsed = [parse(r["Refined_behavior"]) for r in dija]
    q_len = np.array([nt(q) for q, _ in parsed])
    h_len = np.array([nt(h) for h in harmless])
    print(f"[benign] dija question tokens: mean={q_len.mean():.1f} min={q_len.min()} max={q_len.max()}")
    print(f"[benign] harmless pool tokens: mean={h_len.mean():.1f} min={h_len.min()} max={h_len.max()}")

    # Optimal 1-1 length matching. Pool (99) < ids (100), so duplicate it to make the
    # assignment feasible; the cost still minimises total |token-length| mismatch.
    # The duplicate copies carry a large penalty so the solver only reuses a pool item
    # when forced to (exactly once, 100 ids vs 99 items). Without the penalty the solver
    # happily reuses ~43 items for marginally better length fit, which correlates ids and
    # shrinks the effective sample for the paired test.
    pool_idx = np.arange(len(harmless))
    pool_idx = np.concatenate([pool_idx, pool_idx])
    cost = np.abs(q_len[:, None] - h_len[pool_idx][None, :]).astype(float)
    cost[:, len(harmless):] += 1e4
    ri, ci = linear_sum_assignment(cost)
    assign = {int(i): int(pool_idx[j]) for i, j in zip(ri, ci)}
    reuse = len(assign) - len(set(assign.values()))
    raw = np.abs(q_len[:, None] - h_len[pool_idx][None, :])[ri, ci]
    print(f"[benign] question length matching: mean |dtok|={raw.mean():.2f} "
          f"max={raw.max()} | reused pool items={reuse}")

    f_len = np.array([nt(f) for f in FILLER])
    out, stats = [], []
    for i, (r, (q, body)) in enumerate(zip(dija, parsed)):
        bq = harmless[assign[i]]
        lines = [bq]
        # DISTINCT filler per line: DIJA's trailing sentences are all different, so reusing
        # one sentence 4x would itself be a format mismatch. Assign distinct fillers to the
        # 3-4 slots minimising total token-length error. suffix begins with ". ".
        tgt = np.array([nt(s) for _, s in body], dtype=float)
        fc = np.abs((f_len + 2)[None, :] - tgt[:, None])
        _, fi = linear_sum_assignment(fc)
        for (prefix, _), j in zip(body, fi):
            lines.append(f"{prefix}. {FILLER[int(j)]}")
        bs = "\n".join(lines)
        # invariant: mask structure must be byte-identical to the DIJA arm
        assert re.findall(r"<mask:\d+>", bs) == re.findall(r"<mask:\d+>", r["Refined_behavior"]), r["id"]
        out.append({"id": r["id"], "benign_scaffold": bs,
                    "n_inject": sum(int(x) for x in re.findall(r"<mask:(\d+)>", bs)),
                    "source_harmless_idx": assign[i]})
        stats.append((nt(r["Refined_behavior"]), nt(bs)))

    d_tok = np.array([a for a, _ in stats]); b_tok = np.array([b for _, b in stats])
    n_d = np.array([sum(int(x) for x in re.findall(r"<mask:(\d+)>", r["Refined_behavior"])) for r in dija])
    n_b = np.array([o["n_inject"] for o in out])
    print(f"\n[benign] n_inject identical to dija: {int((n_d == n_b).sum())}/100  "
          f"(mean={n_b.mean():.1f} min={n_b.min()} max={n_b.max()})")
    print(f"[benign] scaffold tokens: dija mean={d_tok.mean():.1f}+-{d_tok.std():.1f} | "
          f"benign mean={b_tok.mean():.1f}+-{b_tok.std():.1f}")
    gap = b_tok - d_tok
    print(f"[benign] per-id token gap (benign - dija): mean={gap.mean():+.1f} sd={gap.std():.1f} "
          f"min={gap.min():+d} max={gap.max():+d}")

    p = os.path.join(HELDOUT_DIR, "benign_matched.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n[benign] wrote {p}")
    print("\n--- example (id %s) ---" % out[0]["id"])
    print(out[0]["benign_scaffold"])


if __name__ == "__main__":
    main()
