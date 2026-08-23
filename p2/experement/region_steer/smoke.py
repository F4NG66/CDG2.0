#!/usr/bin/env python3
"""region_steer/smoke.py — STEP 3.1 SMOKE (--limit 2).
Loads LLaDA once, calibrates alpha from baseline L16 injection projection, then runs
E/O/T/R (+Rw) on 2 harm-delivering B cases with IDENTICAL RNG per case (differences are
pure steering, not sampling noise). Dumps before/after text + hook-write confirmation."""
import os, sys, json, argparse
import torch

EXP = "/home/ore99/experement"
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "region_steer"))
sys.path.insert(0, os.path.join(EXP, "dija_attack"))

from ladaAndH import discover_blocks
from runner import RegionSteerRunner, load_injection_direction

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LAYER = 16
POOL = "mean"
CASE_SEED = 20260803          # fixed noise draw shared across arms per case


def load_model(device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}
    model = model.to(device).eval()
    _, blocks = discover_blocks(model)
    return tok, model, blocks


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
        trep = sum(v-1 for v in c.values() if v > 1) / len(tg)
    content = [w for w in re.findall(r"[A-Za-z]{3,}", text)]
    uniq = len(set(content))/len(content) if content else 0.0
    return {"len": len(text), "trigram_rep": round(trep, 3), "uniq_ratio": round(uniq, 3),
            "collapse_heur": bool(content and uniq < 0.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A014", "A001"])
    ap.add_argument("--out", default=os.path.join(EXP, "region_steer", "results", "smoke.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = json.load(open(os.path.join(EXP, "dija_attack/refined_100.json")))
    by_id = {r["id"]: r for r in rows}

    print(f"[smoke] loading {MODEL_ID} bf16 ...", flush=True)
    tok, model, blocks = load_model(args.device)
    v_L, vmeta = load_injection_direction(LAYER, POOL, device=args.device)
    print(f"[smoke] direction: {vmeta['kind']} pool={POOL} layer={LAYER} "
          f"norm={vmeta['norm']:.4f} src={vmeta['path']}", flush=True)
    runner = RegionSteerRunner(model, tok, args.device, blocks, v_L, LAYER)

    fout = open(args.out, "w", encoding="utf-8")

    # ---- 1) CALIBRATION: baseline E per case, read L16 injection projection ----
    print("\n" + "="*80 + "\nCALIBRATION (ARM E baseline, L16 stats)\n" + "="*80, flush=True)
    base = {}
    for cid in args.cases:
        scaffold = by_id[cid]["Refined_behavior"]
        torch.manual_seed(CASE_SEED)
        r = runner.generate_arm(scaffold, "E", 0.0, seed=CASE_SEED)
        tc = r["diag"]["tpl_ctx"]
        base[cid] = r
        print(f"  {cid}: counts={r['region_counts']}  P={r['P']} total={r['total']}", flush=True)
        print(f"        L16 tpl_ctx: mean||h||={tc['mean_norm']:.2f}  mean(h.v)={tc['mean_proj']:.3f}", flush=True)
        cm = collapse_metrics(r["response"])
        print(f"        baseline resp {cm}  resp[:100]={r['response'][:100]!r}", flush=True)

    # alpha sweep: multiples of the natural injection projection at tpl_ctx
    proj_ref = sum(abs(base[c]["diag"]["tpl_ctx"]["mean_proj"]) for c in args.cases)/len(args.cases)
    norm_ref = sum(base[c]["diag"]["tpl_ctx"]["mean_norm"] for c in args.cases)/len(args.cases)
    ALPHAS = [round(2*proj_ref, 2), round(6*proj_ref, 2)]
    print(f"\n[smoke] proj_ref(|mean_proj tpl_ctx|)={proj_ref:.3f}  norm_ref(||h||)={norm_ref:.2f}", flush=True)
    print(f"[smoke] ALPHAS={ALPHAS}  (as multiples 2x,6x of proj_ref; "
          f"alpha/||h|| = {[round(a/norm_ref,3) for a in ALPHAS]})", flush=True)

    # ---- 2) SMOKE arms ----
    print("\n" + "="*80 + "\nSMOKE ARMS (before=E / after=steered)\n" + "="*80, flush=True)
    ARMS = ["O", "T", "R", "Rw"]
    for cid in args.cases:
        scaffold = by_id[cid]["Refined_behavior"]
        e = base[cid]
        rec = {"id": cid, "arm": "E", "alpha": 0.0, "response": e["response"],
               "diag": e["diag"], "region_counts": e["region_counts"],
               "metrics": collapse_metrics(e["response"]), "n_inject": e["n_inject"]}
        fout.write(json.dumps(rec, ensure_ascii=False)+"\n"); fout.flush()
        print(f"\n##### CASE {cid}  (E baseline resp below) #####")
        print(f"  [E]      {e['response'][:220]!r}")
        for arm in ARMS:
            for alpha in ALPHAS:
                torch.manual_seed(CASE_SEED)          # identical noise draw as E
                r = runner.generate_arm(scaffold, arm, alpha, seed=CASE_SEED)
                m = collapse_metrics(r["response"])
                rec = {"id": cid, "arm": arm, "alpha": alpha, "response": r["response"],
                       "diag": r["diag"], "region_counts": r["region_counts"],
                       "metrics": m, "n_inject": r["n_inject"]}
                fout.write(json.dumps(rec, ensure_ascii=False)+"\n"); fout.flush()
                dg = r["diag"]
                print(f"  [{arm} a={alpha:g}] steer_steps={dg['steer_steps']} "
                      f"rows_last={dg['steer_rows_last']} "
                      f"len={m['len']} uniq={m['uniq_ratio']} trep={m['trigram_rep']} "
                      f"collapse={m['collapse_heur']}", flush=True)
                print(f"           resp={r['response'][:220]!r}", flush=True)
    fout.close()
    runner.close()
    print(f"\n[smoke] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
