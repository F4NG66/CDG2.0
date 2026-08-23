#!/usr/bin/env python3
"""study1/kcommit_report.py — FREEZE the surviving Study 1 result: k_commit.

No probe, no GPU, no network. Reads the expansion payloads' aux fields only
(commit_step / region_id / fillable_reply_mask) — never `hidden` — so it runs on
the ragged out_mask-only store directly.

k_i = min{k : i not in M(x_tk)}, over the INJECTED DIJA worksheet blanks
(region_id == R_SCAFFOLD & fillable). Bounded by n_inject, so it cannot balloon
with continuation length (that was the v1 span-locator failure mode).

Two variants are reported:
  ALL      every scaffold blank
  NO-SKEL  scaffold blanks with k_i > --skel-max, dropping the step-0..2
           "structural skeleton" cluster (the worksheet's own connective tokens,
           which the model pins immediately and which carry no harm content)

  python study1/kcommit_report.py --data /scratch/ore99/study1_expansion \
      --judged /scratch/ore99/study1_expansion/judged.jsonl
"""
from __future__ import annotations
import argparse
import json
import os

import torch

R_SCAFFOLD = 0


def med(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def q(v, f):
    if not v:
        return None
    s = sorted(v)
    return float(s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/scratch/ore99/study1_expansion")
    ap.add_argument("--judged", default="/scratch/ore99/study1_expansion/judged.jsonl")
    ap.add_argument("--arm", default="dija")
    ap.add_argument("--skel-max", type=int, default=2,
                    help="k_i <= this is treated as structural skeleton and dropped in NO-SKEL")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    judged = {}
    for line in open(args.judged):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("arm") == args.arm:
            judged[r["id"]] = r

    rows = []
    for cid in sorted(judged):
        p = os.path.join(args.data, f"{cid}__{args.arm}.pt")
        if not os.path.exists(p):
            continue
        pay = torch.load(p, map_location="cpu", weights_only=False)
        commit = pay["commit_step"].long()
        sel = (pay["region_id"] == R_SCAFFOLD) & pay["fillable_reply_mask"]
        k = [int(x) for x in commit[sel].tolist() if x >= 0]
        k_ns = [x for x in k if x > args.skel_max]
        j = judged[cid]
        rows.append({
            "id": cid,
            "harm": bool(j.get("harm_delivered_inclusive")),
            "valence": j.get("valence_category"),
            "n_blank": len(k),
            "n_skel": len(k) - len(k_ns),
            "all": k,
            "nsk": k_ns,
        })

    pos = [r for r in rows if r["harm"]]
    neg = [r for r in rows if not r["harm"]]

    print(f"# k_commit — arm={args.arm}, {len(rows)} cases "
          f"({len(pos)} harm-delivered [valence-inclusive], {len(neg)} not)")
    print(f"# unit = scaffold blank (region_id==R_SCAFFOLD & fillable); steps={args.steps}")
    print(f"# NO-SKEL drops k_i <= {args.skel_max}\n")

    hdr = (f"{'id':6} {'harm':5} {'valence':30} {'blanks':>6} {'skel':>5} "
           f"{'ALL med':>8} {'ALL p25':>8} {'ALL p75':>8} {'NOSKEL med':>10} {'shift':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ma, mn = med(r["all"]), med(r["nsk"])
        sh = "" if (ma is None or mn is None) else f"{mn - ma:+.1f}"
        print(f"{r['id']:6} {str(r['harm']):5} {str(r['valence'])[:30]:30} "
              f"{r['n_blank']:6d} {r['n_skel']:5d} "
              f"{ma if ma is None else f'{ma:8.1f}'} {q(r['all'],.25):8.1f} "
              f"{q(r['all'],.75):8.1f} {mn if mn is None else f'{mn:10.1f}'} {sh:>6}")

    def pooled(rs, key):
        return [x for r in rs for x in r[key]]

    print()
    for name, rs in (("harm-delivered (n=%d)" % len(pos), pos),
                     ("not-delivered  (n=%d)" % len(neg), neg),
                     ("all cases      (n=%d)" % len(rows), rows)):
        pa, pn = pooled(rs, "all"), pooled(rs, "nsk")
        cm_a = [med(r["all"]) for r in rs]
        cm_n = [med(r["nsk"]) for r in rs if r["nsk"]]
        print(f"{name:24} pooled-blank ALL median={med(pa):6.1f} "
              f"[p25 {q(pa,.25):.0f}, p75 {q(pa,.75):.0f}]  n={len(pa):5d}   "
              f"NO-SKEL median={med(pn):6.1f} n={len(pn):5d}")
        print(f"{'':24} median-of-case-medians ALL={med(cm_a):6.1f} "
              f"NO-SKEL={med(cm_n):6.1f}")

    # bimodality of the per-case medians, harm-delivered arm
    cm = sorted((med(r["all"]), r["id"]) for r in pos)
    print("\n# per-case medians, harm-delivered, sorted (bimodality check)")
    print("  " + "  ".join(f"{c:.0f}({i})" for c, i in cm))
    early = [x for x in cm if x[0] < 70]
    late = [x for x in cm if x[0] >= 70]
    print(f"  early cluster (<70): {[i for _, i in early]}  medians {[c for c, _ in early]}")
    print(f"  late  cluster (>=70): n={len(late)}  range "
          f"{min(c for c,_ in late):.0f}-{max(c for c,_ in late):.0f}")

    # which cases move under NO-SKEL
    print(f"\n# cases whose median moves by >=5 steps under NO-SKEL (skel_max={args.skel_max})")
    moved = [(r["id"], med(r["all"]), med(r["nsk"]), r["n_skel"], r["n_blank"])
             for r in rows if r["nsk"] and abs(med(r["nsk"]) - med(r["all"])) >= 5]
    if not moved:
        print("  (none)")
    for i, a, n, ns, nb in moved:
        print(f"  {i}: {a:.1f} -> {n:.1f}  ({ns}/{nb} blanks dropped)")

    # step-0..2 mass
    tot = sum(r["n_blank"] for r in rows)
    sk = sum(r["n_skel"] for r in rows)
    print(f"\n# structural skeleton mass: {sk}/{tot} blanks ({100.0*sk/tot:.1f}%) "
          f"commit at k <= {args.skel_max}")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()
