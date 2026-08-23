#!/usr/bin/env python3
"""PHASE 1 step 1 - build the (prompt, harm_resp, safe_resp) pairing table.

Rule: SAME prompt on both sides, differ only in the response.

Sources
  harm side : graded valence_category == 'unchanged_harmful'
  safe side : graded valence_category == 'substantive_safe_substitution'
              (real LLaDA text, same scaffold, same length band)
              -> gaps filled later by generate_safe.py (DeepSeek)

Both pools get the SAME degeneracy filter. Filtering only one side would make
the filter itself the signal.

The prefill arm leaves a textual watermark inside the response ("First, you",
"Next, you", ...), so harm/safe arm marginals are balanced by a greedy swap pass.

Read-only w.r.t. everything outside harm_dir/.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics

from textnorm import quality, strip_echo, strip_tail

PREFILL_RUNS = "/home/ore99/serverFiles/dijawithprefill/runs/full"
PROMPTS = "/home/ore99/serverFiles/prompts/cdg_injection/B_harmful_injected/cases.json"
OUTDIR = "/home/ore99/experement/harm_dir/data"

HARM_CAT = "unchanged_harmful"
SAFE_CAT = "substantive_safe_substitution"

# degeneracy thresholds - applied identically to harm and safe
MAX_DIGIT_FRAC = 0.15     # kills "123456789012345" slot filler
MIN_UNIQ_RATIO = 0.45     # kills verbatim repetition loops
MAX_TRIGRAM_REP = 0.25
MIN_WORDS = 40


def jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def passes(q):
    return (q["words"] >= MIN_WORDS
            and q["digit_frac"] <= MAX_DIGIT_FRAC
            and q["uniq_word_ratio"] >= MIN_UNIQ_RATIO
            and q["trigram_rep"] <= MAX_TRIGRAM_REP)


def load_pool():
    behaviors = {c["id"]: c for c in json.load(open(PROMPTS))}
    arms = sorted(d for d in glob.glob(PREFILL_RUNS + "/*/")
                  if os.path.exists(d + "graded_judge.jsonl"))
    recs = []
    for a in arms:
        arm = a.rstrip("/").split("/")[-1]
        graded = {r["case_id"]: r for r in jsonl(a + "graded_judge.jsonl") if r.get("group") == "B"}
        man = {r["case_id"]: r for r in jsonl(a + "manifest.jsonl")
               if r["variant"] == "B_harmful_injected"}
        for cid, g in graded.items():
            cat = g["graded"]["valence_category"]
            if cat not in (HARM_CAT, SAFE_CAT):
                continue
            case = behaviors[cid]
            # Judge degeneracy on the model's full output, but STORE the
            # tail-stripped text. Scoring post-strip would inflate digit_frac
            # (the canned tail is ~10 words of clean prose that dilutes it) and
            # over-drop otherwise usable responses.
            full = strip_echo(man[cid]["response_text"], case["behavior"])
            text = strip_tail(full)
            q = quality(full)
            q["words"] = len(text.split())          # length band uses stored text
            recs.append({
                "case_id": cid, "arm": arm, "side": "harm" if cat == HARM_CAT else "safe",
                "category": cat, "text": text,
                "valence_score": g["graded"].get("valence_score"),
                "specificity": g["graded"].get("specificity_score"),
                "collapse": g["graded"].get("collapse_score"),
                "source": "llada_prefill_arm",
                **q,
                "kept": passes(q),
            })
    return behaviors, recs


def balance_arms(pairs, harm_by_id, safe_by_id, rounds=200):
    """Greedy swaps that reduce |harm_arm_count - safe_arm_count| without
    breaking the +-20% length band."""
    # Only natural pairs can be balanced: a generated safe response has no arm,
    # so counting the 50 generated rows on the harm side against an empty safe
    # side makes the objective unreachable and the swaps no-ops.
    def marginals():
        nat = [p for p in pairs if p["safe_source"] == "llada_prefill_arm"]
        h = collections.Counter(p["harm_arm"] for p in nat)
        s = collections.Counter(p["safe_arm"] for p in nat)
        return h, s

    def cost():
        h, s = marginals()
        return sum(abs(h.get(a, 0) - s.get(a, 0)) for a in set(h) | set(s))

    def inband(hw, sw):
        return hw > 0 and sw > 0 and abs(math.log(sw / hw)) <= math.log(1.20)

    best = cost()
    for _ in range(rounds):
        improved = False
        for p in pairs:
            cid = p["case_id"]
            for alt in harm_by_id.get(cid, []):
                if alt["arm"] == p["harm_arm"]:
                    continue
                if p["safe_source"] == "llada_prefill_arm" and not inband(alt["words"], p["safe_words"]):
                    continue
                old = (p["harm_arm"], p["harm_text"], p["harm_words"], p["harm_specificity"])
                p["harm_arm"], p["harm_text"] = alt["arm"], alt["text"]
                p["harm_words"], p["harm_specificity"] = alt["words"], alt["specificity"]
                c = cost()
                if c < best:
                    best, improved = c, True
                else:
                    (p["harm_arm"], p["harm_text"],
                     p["harm_words"], p["harm_specificity"]) = old
            if p["safe_source"] != "llada_prefill_arm":
                continue
            for alt in safe_by_id.get(cid, []):
                if alt["arm"] == p["safe_arm"] or not inband(p["harm_words"], alt["words"]):
                    continue
                old = (p["safe_arm"], p["safe_text"], p["safe_words"], p["safe_specificity"])
                p["safe_arm"], p["safe_text"] = alt["arm"], alt["text"]
                p["safe_words"], p["safe_specificity"] = alt["words"], alt["specificity"]
                c = cost()
                if c < best:
                    best, improved = c, True
                else:
                    (p["safe_arm"], p["safe_text"],
                     p["safe_words"], p["safe_specificity"]) = old
        if not improved:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: keep only N base ids")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "pairs.jsonl"))
    args = ap.parse_args()

    behaviors, recs = load_pool()

    print("=" * 74)
    print("DEGENERACY FILTER (identical thresholds on both sides)")
    print("=" * 74)
    print(f"  words>={MIN_WORDS}  digit_frac<={MAX_DIGIT_FRAC}  "
          f"uniq_word_ratio>={MIN_UNIQ_RATIO}  trigram_rep<={MAX_TRIGRAM_REP}")
    for side in ("harm", "safe"):
        r = [x for x in recs if x["side"] == side]
        k = [x for x in r if x["kept"]]
        drop = collections.Counter()
        for x in r:
            if x["kept"]:
                continue
            if x["words"] < MIN_WORDS: drop["too_short"] += 1
            elif x["digit_frac"] > MAX_DIGIT_FRAC: drop["digit_filler"] += 1
            elif x["uniq_word_ratio"] < MIN_UNIQ_RATIO: drop["repetition_loop"] += 1
            else: drop["trigram_rep"] += 1
        print(f"  {side:4s}: kept {len(k):3d}/{len(r):3d}   dropped {dict(drop)}")

    kept = [x for x in recs if x["kept"]]
    harm_by_id = collections.defaultdict(list)
    safe_by_id = collections.defaultdict(list)
    for x in kept:
        (harm_by_id if x["side"] == "harm" else safe_by_id)[x["case_id"]].append(x)

    # prefer the most specific harmful response, the least specific safe one
    for d in harm_by_id.values():
        d.sort(key=lambda r: -(r["specificity"] or 0))
    for d in safe_by_id.values():
        d.sort(key=lambda r: (r["specificity"] or 0))

    ids = sorted(harm_by_id)
    if args.limit:
        ids = ids[:args.limit]

    pairs = []
    for cid in ids:
        h = harm_by_id[cid][0]
        cands = safe_by_id.get(cid, [])
        s = min(cands, key=lambda r: abs(math.log(r["words"] / h["words"]))) if cands else None
        pairs.append({
            "case_id": cid,
            "behavior": behaviors[cid]["behavior"],
            "user_content": behaviors[cid]["user_content"],
            "harm_arm": h["arm"], "harm_text": h["text"], "harm_words": h["words"],
            "harm_specificity": h["specificity"], "harm_source": "llada_prefill_arm",
            "safe_arm": s["arm"] if s else None,
            "safe_text": s["text"] if s else None,
            "safe_words": s["words"] if s else None,
            "safe_specificity": s["specificity"] if s else None,
            "safe_source": "llada_prefill_arm" if s else "NEEDS_GENERATION",
        })

    imb = balance_arms(pairs, harm_by_id, safe_by_id)

    print()
    print("=" * 74)
    print("PAIRING TABLE")
    print("=" * 74)
    nat = [p for p in pairs if p["safe_source"] == "llada_prefill_arm"]
    gap = [p for p in pairs if p["safe_source"] == "NEEDS_GENERATION"]
    print(f"  base ids with a usable harmful response : {len(pairs)}")
    print(f"    natural safe pair (both sides LLaDA)  : {len(nat)}")
    print(f"    needs generated safe response         : {len(gap)}")

    if nat:
        ratios = [p["safe_words"] / p["harm_words"] for p in nat]
        inb = sum(1 for r in ratios if 0.8 <= r <= 1.2)
        print(f"\n  natural pairs, safe/harm word ratio: "
              f"median {statistics.median(ratios):.2f}  in +-20% band {inb}/{len(nat)}")
        print(f"  harm words median {statistics.median([p['harm_words'] for p in nat]):.0f}  "
              f"safe words median {statistics.median([p['safe_words'] for p in nat]):.0f}")

    hn = collections.Counter(p["harm_arm"] for p in nat)
    sn = collections.Counter(p["safe_arm"] for p in nat)
    hg = collections.Counter(p["harm_arm"] for p in gap)
    print(f"\n  arm marginals (natural pairs balanced; residual imbalance = {imb}):")
    print(f"    {'arm':10s} {'nat_harm':>9s} {'nat_safe':>9s} {'gen_harm':>9s}")
    for a in sorted(set(hn) | set(sn) | set(hg)):
        print(f"    {a:10s} {hn.get(a,0):9d} {sn.get(a,0):9d} {hg.get(a,0):9d}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nwrote {args.out}  ({len(pairs)} rows, {len(gap)} awaiting generation)")


if __name__ == "__main__":
    main()
