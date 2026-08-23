#!/usr/bin/env python3
"""clockv2/build_benign_op.py — THIRD ARM: operational, first-person BENIGN scaffold.

Same construction as build_benign.py (byte-identical <mask:N>, verbatim body-line prefixes,
<<TPL>> wrapper via PaperRunner, length-matched trailing filler), but the question line is
drawn from benign_operational_pool.BENIGN_OPERATIONAL -- first-person, operational
("How can I safely ..."), which the second arm was not.

WHY: arm 2's benign pool was 99/99 "What are the ..." and 0% first-person, while the DIJA
arm is 87% first-person / 82% "How can I". So arm 2's surviving dija-benign signal could
not be attributed to harm rather than operational-vs-informational STANCE. This arm matches
stance and leaves harm as the contrast.

LEAKAGE: the pool is authored (see benign_operational_pool.py for why no drawable pool
exists) and is verified here to have ZERO exact overlap and bounded Jaccard against EVERY
probe's fitting set:
    v_refusal      <- ho['harmful'] (43) and ho['harmless'] (99)
    v_injection_svd<- ho['harmful'] (43)
    tau_bank       <- D_neutral_clean neutral seeds
    (plus the 100 eval behaviors, which are not a fitting set but must not leak either)

NO HARMFUL TEXT IS AUTHORED. No model is loaded here; construction is pure string work.

    python clockv2/build_benign_op.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer

from benign_operational_pool import BENIGN_OPERATIONAL
from build_benign import FILLER, parse
from common import CASES_A, HELDOUT_DIR, MODEL_ID, REFINED_100, jaccard

D_NEUTRAL = "/home/ore99/serverFiles/prompts/cdg_injection/D_neutral_clean/cases.json"
UNSAFE = "/home/ore99/experement/unsafe.jsonl"


def leak_check(pool):
    """Zero exact overlap + max Jaccard vs every probe fitting set. Returns ok, report."""
    ho = json.load(open(os.path.join(HELDOUT_DIR, "contrastive.json")))
    sets = {
        "v_refusal harmful (43)": ho["harmful"],
        "v_refusal harmless (99)": ho["harmless"],
        "v_injection_svd fit (43)": ho["harmful"],
        "tau_bank pool D_neutral_clean (100)": [d["behavior"] for d in json.load(open(D_NEUTRAL))],
        "unsafe.jsonl superset (50)": [json.loads(l)["question"] for l in open(UNSAFE) if l.strip()],
        "eval behaviors (100, not a fit set)": [c["behavior"] for c in json.load(open(CASES_A))],
    }
    rep, ok = [], True
    for name, ref in sets.items():
        exact = len(set(pool) & set(ref))
        mj, arg = 0.0, None
        for p in pool:
            for r in ref:
                j = jaccard(p, r)
                if j > mj:
                    mj, arg = j, (p, r)
        ok &= (exact == 0)
        rep.append((name, len(ref), exact, mj, arg))
    return ok, rep


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    nt = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    pool = BENIGN_OPERATIONAL
    assert len(set(pool)) == len(pool), "authored pool has duplicate items"

    print("=" * 96)
    print("LEAKAGE CHECK — authored operational-benign pool vs EVERY probe fitting set")
    print("=" * 96)
    ok, rep = leak_check(pool)
    print(f"{'fitting set':>38} | {'n':>4} {'exact':>6} {'maxJ':>6} | closest pair")
    for name, nref, exact, mj, arg in rep:
        s = f"{arg[0][:34]}... ~ {arg[1][:34]}..." if arg else ""
        print(f"{name:>38} | {nref:>4} {exact:>6} {mj:>6.3f} | {s}")
    print(f"\n  zero exact overlap everywhere: {ok}")
    print(f"  max Jaccard across all fitting sets: {max(r[3] for r in rep):.3f} "
          f"(build_heldout used J<=0.4 as its near-dup threshold)")
    if not ok:
        raise SystemExit("LEAKAGE: authored pool overlaps a fitting set")

    dija = json.load(open(REFINED_100))
    parsed = [parse(r["Refined_behavior"]) for r in dija]
    q_len = np.array([nt(q) for q, _ in parsed])
    p_len = np.array([nt(s) for s in pool])
    print(f"\n[op] dija question tokens: mean={q_len.mean():.1f} min={q_len.min()} max={q_len.max()}")
    print(f"[op] authored pool tokens: mean={p_len.mean():.1f} min={p_len.min()} max={p_len.max()}")

    # length matching; duplicate copies penalised so reuse only happens when forced
    reps = int(np.ceil(len(dija) / len(pool)))
    pidx = np.concatenate([np.arange(len(pool))] * (reps + 1))
    cost = np.abs(q_len[:, None] - p_len[pidx][None, :]).astype(float)
    for k in range(1, reps + 1):
        cost[:, k * len(pool):(k + 1) * len(pool)] += 1e4 * k
    ri, ci = linear_sum_assignment(cost)
    assign = {int(i): int(pidx[j]) for i, j in zip(ri, ci)}
    raw = np.abs(q_len[:, None] - p_len[pidx][None, :])[ri, ci]
    print(f"[op] question length matching: mean |dtok|={raw.mean():.2f} max={raw.max()} | "
          f"reused items={len(assign)-len(set(assign.values()))} (forced: 100 ids vs {len(pool)} pool)")

    f_len = np.array([nt(f) for f in FILLER])
    out, stats = [], []
    for i, (r, (q, body)) in enumerate(zip(dija, parsed)):
        lines = [pool[assign[i]]]
        tgt = np.array([nt(s) for _, s in body], dtype=float)
        fc = np.abs((f_len + 2)[None, :] - tgt[:, None])
        _, fi = linear_sum_assignment(fc)
        for (prefix, _), j in zip(body, fi):
            lines.append(f"{prefix}. {FILLER[int(j)]}")
        bs = "\n".join(lines)
        import re
        assert re.findall(r"<mask:\d+>", bs) == re.findall(r"<mask:\d+>", r["Refined_behavior"]), r["id"]
        out.append({"id": r["id"], "benign_scaffold": bs,
                    "n_inject": sum(int(x) for x in re.findall(r"<mask:(\d+)>", bs)),
                    "source_pool_idx": assign[i]})
        stats.append((nt(r["Refined_behavior"]), nt(bs)))

    import re
    n_d = np.array([sum(int(x) for x in re.findall(r"<mask:(\d+)>", r["Refined_behavior"])) for r in dija])
    n_b = np.array([o["n_inject"] for o in out])
    d_tok = np.array([a for a, _ in stats]); b_tok = np.array([b for _, b in stats])
    gap = b_tok - d_tok
    print(f"\n[op] n_inject identical to dija: {int((n_d==n_b).sum())}/100 (mean={n_b.mean():.1f})")
    print(f"[op] scaffold tokens: dija {d_tok.mean():.1f}+-{d_tok.std():.1f} | "
          f"op-benign {b_tok.mean():.1f}+-{b_tok.std():.1f}")
    print(f"[op] per-id token gap (op-benign - dija): mean={gap.mean():+.1f} sd={gap.std():.1f} "
          f"min={gap.min():+d} max={gap.max():+d}")

    import collections
    fp = sum(bool(re.search(r"\b(I|my|me)\b", o["benign_scaffold"].split("\n")[0])) for o in out)
    op = collections.Counter(" ".join(o["benign_scaffold"].split()[:3]).lower() for o in out)
    print(f"\n[op] REGISTER MATCH (the point of this arm):")
    print(f"     first-person questions: {fp}/100  (dija arm: 87/100 | arm-2 benign: 0/100)")
    print(f"     top opener: {op.most_common(1)[0]}")

    p = os.path.join(HELDOUT_DIR, "benign_op_matched.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n[op] wrote {p}")
    print(f"\n--- example (id {out[0]['id']}) ---\n{out[0]['benign_scaffold']}")


if __name__ == "__main__":
    main()
