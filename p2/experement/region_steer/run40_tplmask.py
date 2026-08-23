#!/usr/bin/env python3
"""region_steer/run40_tplmask.py — LAST ARM: tpl_mask (TM) added to the EXACT n=40 design.

Identical to run40.py (same 40 cases = 20 strict + 20 disclaimer, same alphas {21,42,84,126},
same per-case RNG seed=CASE_SEED0+case#, same direction/layer/block/steps, -v de-inject) with
ONE change: arm TM (steer the MASKED BLANKS inside the injected template span, tpl_mask) is added
to B AND to C. TM is the structural complement of T (tpl_ctx) within the same span, so the run
yields the DIRECT TM-vs-T contrast at matched alpha plus TM-vs-E and TM selectivity on benign C.

  B arms: E / T / O / R / Rw / TM   (21 traj/case: E + 5 arms * 4 alphas)
  C arms: E / T / O / TM            (13 traj/case: E + 3 arms * 4 alphas)
  total : 34 traj/case * 40 = 1360.
Nothing in dija_attack/ or clockv2/ is edited; runner.py's TM branch is the only new code path."""
import os, sys, json, argparse
import torch

EXP = "/home/ore99/experement"
sys.path[:0] = [EXP, os.path.join(EXP, "region_steer"), os.path.join(EXP, "dija_attack")]
from ladaAndH import discover_blocks
from runner import RegionSteerRunner, load_injection_direction

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LAYER, POOL = 16, "mean"
ALPHAS = [21.0, 42.0, 84.0, 126.0]
CASE_SEED0 = 20260803
B_ARMS = ["T", "O", "R", "Rw", "TM"]     # + TM
C_ARMS = ["T", "O", "TM"]                 # + TM (selectivity control)
STRICT20 = ["A004","A007","A012","A014","A016","A021","A022","A026","A027","A031",
            "A032","A033","A037","A039","A040","A044","A045","A046","A055","A059"]
DISC20 = ["A000","A001","A003","A006","A008","A010","A011","A013","A015","A017",
          "A018","A019","A020","A023","A024","A025","A028","A029","A034","A035"]
STRATUM = {**{c: "strict" for c in STRICT20}, **{c: "disclaimer" for c in DISC20}}
CASES = sorted(STRICT20 + DISC20)


def collapse_metrics(text):
    import re
    from collections import defaultdict
    w = text.split()
    if len(w) < 4:
        trep = 0.0
    else:
        tg = [tuple(w[i:i+3]) for i in range(len(w)-2)]
        c = defaultdict(int)
        for t in tg: c[t] += 1
        trep = sum(v-1 for v in c.values() if v > 1)/len(tg)
    content = [x for x in re.findall(r"[A-Za-z]{3,}", text)]
    uniq = len(set(content))/len(content) if content else 0.0
    return {"len": len(text), "trigram_rep": round(trep, 4), "uniq_ratio": round(uniq, 4),
            "collapse_heur": bool(content and uniq < 0.5)}


def done_traj(path):
    d = set()
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if l:
                try: d.add(json.loads(l)["traj_id"])
                except Exception: pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP, "region_steer/batch_tplmask/gen.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Bm = {r["id"]: r for r in json.load(open(os.path.join(EXP, "dija_attack/refined_100.json")))}
    Cm = {r["id"]: r for r in json.load(open(os.path.join(EXP, "clockv2/data/benign_op_matched.json")))}

    print(f"[run40tm] loading {MODEL_ID} bf16 ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}
    model = model.to(args.device).eval()
    _, blocks = discover_blocks(model)
    v_L, vmeta = load_injection_direction(LAYER, POOL, device=args.device)
    print(f"[run40tm] dir {vmeta['kind']} L{LAYER} norm={vmeta['norm']:.4f}; cases={len(CASES)}", flush=True)
    runner = RegionSteerRunner(model, tok, args.device, blocks, v_L, LAYER)

    jobs = []
    for cid in CASES:
        seed = CASE_SEED0 + int(cid[1:])
        b_scaf, b_beh = Bm[cid]["Refined_behavior"], Bm[cid]["behavior"]
        c_scaf = Cm[cid]["benign_scaffold"]
        jobs.append(("B", cid, "E", 0.0, seed, b_scaf, b_beh))
        for arm in B_ARMS:
            for a in ALPHAS: jobs.append(("B", cid, arm, a, seed, b_scaf, b_beh))
        jobs.append(("C", cid, "E", 0.0, seed, c_scaf, ""))
        for arm in C_ARMS:
            for a in ALPHAS: jobs.append(("C", cid, arm, a, seed, c_scaf, ""))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = done_traj(args.out)
    fout = open(args.out, "a", encoding="utf-8")
    print(f"[run40tm] {len(jobs)} jobs; {len(done)} done", flush=True)
    n = 0
    for grp, cid, arm, alpha, seed, scaffold, beh in jobs:
        tid = f"{grp}_{cid}_{arm}_a{alpha:g}"
        if tid in done: continue
        torch.manual_seed(seed)
        r = runner.generate_arm(scaffold, arm, alpha, seed=seed)
        row = {"traj_id": tid, "group": grp, "id": cid, "stratum": STRATUM[cid], "arm": arm,
               "alpha": alpha, "seed": seed, "behavior": beh, "response": r["response"],
               "tpl_filled": r["tpl_filled"], "n_inject": r["n_inject"],
               "region_counts": r["region_counts"], "diag": r["diag"],
               "metrics": collapse_metrics(r["response"])}
        fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 25 == 0:
            print(f"[run40tm] {n} new (last {tid} len={row['metrics']['len']})", flush=True)
    fout.close(); runner.close()
    print(f"[run40tm] wrote {n} new rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
