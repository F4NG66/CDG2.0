#!/usr/bin/env python3
"""PHASE 2'' step 2 - assemble the VALID Gate E transfer target (LLaDA both sides).

harm = LLaDA DIJA-jailbroken completion (B_harmful_injected, ASR=comply) - LLaDA's
       own actionable output, the thing we ultimately want to steer away from.
safe = LLaDA-authored matched-length safe response (generate_llada_safe.py).

Both sides LLaDA, so this is a within-LLaDA harm/safe contrast - exactly the axis
v_harm_ds must transfer onto to be useful for steering. Filters:
  - Jaccard(harm, safe) <= 0.35     real content contrast (not the Phase 2 trap)
  - length within +/-20%            kills the 12.5x length confound of the free
                                    A-refusal arm
  - safe arm non-degenerate         trigram_rep < 0.35 and >= 30 words, and it must
                                    actually decline (not a second harm arm)

Also emits the FREE natural target (harm=B-comply, safe=A bare-behaviour refusal)
for side-by-side comparison, so the length confound there is on the record.

Outputs:
  data/pairs_gateE.jsonl        matched-length LLaDA harm/safe pairs (primary)
  data/pairs_gateE_natural.jsonl  free A/B pairs (length-confounded, comparison)
"""
import argparse
import json
import os
import re
import statistics

from textnorm import quality

DATA = "/home/ore99/experement/harm_dir/data"
BASE_MANIFEST = "/home/ore99/serverFiles/outputs/manifest.jsonl"
JACCARD_MAX = 0.35
# LLaDA finishes its refusal and emits EOS, so the generated safe arm lands ~15%
# short of the harm arm's length; +/-25% keeps those without forcing degenerate
# padding. Still a small length gap vs the 12.5x of the free A-refusal arm, and
# the analyzer reports length diagnostics on top.
LEN_TOL = 0.25
MIN_WORDS = 30
MAX_TRIGRAM_REP = 0.35

REFUSE = re.compile(
    r"\b(i'?m sorry|i can'?t|i cannot|i am unable|i won'?t|cannot assist|can'?t help|"
    r"can'?t assist|unable to help|i must decline|not able to|do not|don'?t|"
    r"consult|professional|healthcare|doctor|physician|instead|unsafe|dangerous|"
    r"seek|licensed|qualified|emergency)\b", re.I)


def ws(t):
    return set(re.findall(r"[a-z0-9]+", (t or "").lower()))


def jac(a, b):
    A, B = ws(a), ws(b)
    return len(A & B) / max(1, len(A | B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs_ds.jsonl"))
    ap.add_argument("--llada-safe", default=os.path.join(DATA, "llada_safe.jsonl"),
                    help="comma-separated safe-arm caches; per case the arm with "
                         "len_ratio closest to 1.0 (harm-agnostic) that passes the "
                         "content/quality filters is chosen")
    ap.add_argument("--out", default=os.path.join(DATA, "pairs_gateE.jsonl"))
    ap.add_argument("--out-natural", default=os.path.join(DATA, "pairs_gateE_natural.jsonl"))
    args = ap.parse_args()

    behav = {json.loads(l)["case_id"]: json.loads(l)["behavior"] for l in open(args.pairs)}
    uc = {json.loads(l)["case_id"]: json.loads(l)["user_content"] for l in open(args.pairs)}
    man = [json.loads(l) for l in open(BASE_MANIFEST)]
    Bm = {m["case_id"]: m for m in man if m["variant"] == "B_harmful_injected"}
    Am = {"B" + m["case_id"][1:]: m for m in man if m["variant"] == "A_harmful_clean"}

    # gather every candidate safe arm per case across the provided caches
    cands = {}
    for path in args.llada_safe.split(","):
        for l in open(path.strip()):
            r = json.loads(l)
            cands.setdefault(r["case_id"], []).append(r["safe_text_llada"])

    def assess(cid, safe):
        harm = Bm[cid]["response_text"] or ""
        hw, sw = len(harm.split()), len(safe.split())
        j = round(jac(harm, safe), 4)
        q = quality(safe)
        lr = sw / max(1, hw)
        reason = None
        if j > JACCARD_MAX:
            reason = f"jaccard {j:.2f}>{JACCARD_MAX}"
        elif not (1 - LEN_TOL <= lr <= 1 + LEN_TOL):
            reason = f"len_ratio {lr:.2f} outside +/-{LEN_TOL}"
        elif sw < MIN_WORDS:
            reason = f"safe too short {sw}w"
        elif q["trigram_rep"] > MAX_TRIGRAM_REP:
            reason = f"degenerate trigram_rep {q['trigram_rep']:.2f}"
        elif len(REFUSE.findall(safe)) < 2:
            reason = "safe not clearly a refusal/redirect"
        rec = {"case_id": cid, "behavior": behav[cid], "user_content": uc[cid],
               "author": "llada", "harm_text": harm, "harm_words": hw,
               "safe_text": safe, "safe_words": sw, "len_ratio": round(lr, 3),
               "jaccard": j, "safe_trigram_rep": q["trigram_rep"]}
        return rec, reason

    kept, dropped = [], []
    for cid in sorted(cands):
        assessed = [assess(cid, s) for s in cands[cid]]
        passing = [(rec, None) for rec, why in assessed if why is None]
        if passing:
            # harm-agnostic pick: len_ratio closest to 1.0 among passing candidates
            kept.append(min(passing, key=lambda rw: abs(rw[0]["len_ratio"] - 1.0)))
        else:
            # report the closest-to-band candidate's reason for the drop log
            dropped.append(min(assessed, key=lambda rw: abs(rw[0]["len_ratio"] - 1.0)))

    kept_rows = [r for r, _ in kept]
    print("=" * 78)
    print("GATE E MATCHED-LENGTH TARGET (harm=LLaDA-DIJA, safe=LLaDA-generated)")
    print("=" * 78)
    print(f"  cases with candidate arms : {len(cands)}")
    print(f"  SURVIVORS                : {len(kept_rows)}")
    print(f"  dropped                  : {len(dropped)}")
    if kept_rows:
        jj = [r["jaccard"] for r in kept_rows]
        lr = [r["len_ratio"] for r in kept_rows]
        print(f"  Jaccard(harm,safe) median {statistics.median(jj):.3f} "
              f"(min {min(jj):.3f} max {max(jj):.3f})")
        print(f"  len_ratio safe/harm median {statistics.median(lr):.3f} "
              f"(min {min(lr):.3f} max {max(lr):.3f})")
        print(f"  harm words median {statistics.median(r['harm_words'] for r in kept_rows):.0f}  "
              f"safe words median {statistics.median(r['safe_words'] for r in kept_rows):.0f}")
    if dropped:
        print("\n  dropped:")
        for r, why in dropped:
            print(f"    {r['case_id']}  {why}  (harm {r['harm_words']}w safe {r['safe_words']}w "
                  f"J={r['jaccard']:.2f})")

    with open(args.out, "w") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(kept_rows)} matched pairs -> {args.out}")

    # ---- free natural target (length-confounded, for comparison) ----
    nat = []
    for cid in sorted(cands):
        if not (Bm.get(cid) and Am.get(cid)):
            continue
        if not (Bm[cid].get("judge") and Bm[cid]["judge"]["label"] == "comply"):
            continue
        harm = Bm[cid]["response_text"] or ""
        safe = Am[cid]["response_text"] or ""
        nat.append({"case_id": cid, "behavior": behav[cid], "user_content": uc[cid],
                    "author": "llada", "harm_text": harm, "harm_words": len(harm.split()),
                    "safe_text": safe, "safe_words": len(safe.split()),
                    "len_ratio": round(len(safe.split()) / max(1, len(harm.split())), 3),
                    "jaccard": round(jac(harm, safe), 4)})
    with open(args.out_natural, "w") as f:
        for r in nat:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(nat)} natural (length-confounded) pairs -> {args.out_natural}")
    if nat:
        print(f"  natural len_ratio median {statistics.median(r['len_ratio'] for r in nat):.3f}  "
              f"(this is the 12.5x confound we are avoiding in the matched set)")


if __name__ == "__main__":
    main()
