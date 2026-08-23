#!/usr/bin/env python3
"""region_steer/run_arms.py — STEP 3.2 limit-10 falsification run (generation only, offline).

B (harmful-injected): 10 harm-delivering cases x {E, O, T, R, Rw} x alpha{21,42,84}  = 130
C (benign-injected) : same 10 IDs        x {E, O, T}          x alpha{21,42,84}  =  70
Same direction (-v, de-inject) for every steered arm; arms differ only in WHERE v is applied.
Identical RNG per case across arms (differences = pure steering). Resumable by traj_id.
Judging is a separate step (judge_arms.py)."""
import os, sys, json, argparse
import torch

EXP = "/home/ore99/experement"
sys.path[:0] = [EXP, os.path.join(EXP, "region_steer"), os.path.join(EXP, "dija_attack")]
from ladaAndH import discover_blocks
from runner import RegionSteerRunner, load_injection_direction

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LAYER, POOL = 16, "mean"
ALPHAS = [21.0, 42.0, 84.0]
CASE_SEED0 = 20260803
B_ARMS = ["O", "T", "R", "Rw"]
C_ARMS = ["O", "T"]
CASES = ["A000", "A001", "A003", "A004", "A005", "A006", "A007", "A008", "A009", "A010"]


def collapse_metrics(text):
    import re
    from collections import defaultdict
    words = text.split()
    if len(words) < 4:
        trep = 0.0
    else:
        tg = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        c = defaultdict(int)
        for t in tg: c[t] += 1
        trep = sum(v-1 for v in c.values() if v > 1)/len(tg)
    content = [w for w in re.findall(r"[A-Za-z]{3,}", text)]
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


def load_model(device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}
    model = model.to(device).eval()
    _, blocks = discover_blocks(model)
    return tok, model, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP, "region_steer/results/gen.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    B_meta = {r["id"]: r for r in json.load(open(os.path.join(EXP, "dija_attack/refined_100.json")))}
    C_meta = {r["id"]: r for r in json.load(open(os.path.join(EXP, "clockv2/data/benign_op_matched.json")))}

    print(f"[run] loading {MODEL_ID} bf16 ...", flush=True)
    tok, model, blocks = load_model(args.device)
    v_L, vmeta = load_injection_direction(LAYER, POOL, device=args.device)
    print(f"[run] direction {vmeta['kind']} pool={POOL} L{LAYER} norm={vmeta['norm']:.4f}", flush=True)
    runner = RegionSteerRunner(model, tok, args.device, blocks, v_L, LAYER)

    # build job list: (group, cid, arm, alpha, seed, scaffold, behavior)
    jobs = []
    for ci, cid in enumerate(CASES):
        seed = CASE_SEED0 + ci
        b_scaf = B_meta[cid]["Refined_behavior"]; b_beh = B_meta[cid]["behavior"]
        c_scaf = C_meta[cid]["benign_scaffold"]
        jobs.append(("B", cid, "E", 0.0, seed, b_scaf, b_beh))
        for arm in B_ARMS:
            for a in ALPHAS:
                jobs.append(("B", cid, arm, a, seed, b_scaf, b_beh))
        jobs.append(("C", cid, "E", 0.0, seed, c_scaf, ""))
        for arm in C_ARMS:
            for a in ALPHAS:
                jobs.append(("C", cid, arm, a, seed, c_scaf, ""))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = done_traj(args.out)
    fout = open(args.out, "a", encoding="utf-8")
    print(f"[run] {len(jobs)} jobs; {len(done)} already done", flush=True)
    n = 0
    for grp, cid, arm, alpha, seed, scaffold, beh in jobs:
        tid = f"{grp}_{cid}_{arm}_a{alpha:g}"
        if tid in done:
            continue
        torch.manual_seed(seed)                       # identical noise draw per case
        r = runner.generate_arm(scaffold, arm, alpha, seed=seed)
        row = {"traj_id": tid, "group": grp, "id": cid, "arm": arm, "alpha": alpha,
               "seed": seed, "behavior": beh, "response": r["response"],
               "tpl_filled": r["tpl_filled"], "n_inject": r["n_inject"],
               "region_counts": r["region_counts"], "diag": r["diag"],
               "metrics": collapse_metrics(r["response"])}
        fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
        n += 1
        if n % 10 == 0:
            print(f"[run] {n} new gens (last {tid}: len={row['metrics']['len']} "
                  f"uniq={row['metrics']['uniq_ratio']})", flush=True)
    fout.close(); runner.close()
    print(f"[run] wrote {n} new rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
