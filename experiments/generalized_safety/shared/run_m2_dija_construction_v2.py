#!/usr/bin/env python3
"""M2 DIJA construction executor v2 (smoke + confirmatory) - seed policy v2.

Identical to run_m2_dija_construction_v1.py except: imports seed policy v2 and applies the
frozen deterministic runtime (configure_deterministic_runtime) before the model is loaded,
and records/asserts it. Sampling configuration is unchanged.

Reuses M1's official DIJA refiner VERBATIM: imports Refiner from the SHA-verified
DIJA checkout (commit 56ba341) and calls Refiner.qwen_generate(prompt, 200) -
the exact call M1's run_refinement_hf made, inside torch.no_grad(), serially,
batch size 1, with the model's own generation_config (do_sample=True, T=0.7,
top_p=0.8, top_k=20, repetition_penalty=1.05).

Additions (authorized, INC-4 decision):
  - seed policy: apply_seed(derive_seed(call_id)) immediately before each call
  - fresh-source input adapter (harmful_source.request -> B, benign_source.request -> C),
    string identifiers, M2 output paths
  - a pass-through wrapper on model.generate that records output token ids
    (does not alter arguments or results)

No rerolls. Completed call_ids are never regenerated (resume only fills missing).
LLaDA is never loaded or invoked.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m2_dija_seed_policy_v2 as SP  # noqa: E402

DIJA_ROOT = Path(__import__("os").environ["CDG_DIJA_ROOT"])
REFINE_DIR = DIJA_ROOT / "run_harmbench/refine_prompt"
MODEL = DIJA_ROOT / "hf_models/Qwen2.5-7B-Instruct"
TEMPLATE = REFINE_DIR / "redteam_prompt_template.txt"
MAX_NEW_TOKENS = 200
EXPECTED_COMMIT = "56ba341b3377423e6da212013aec6e825710a64c"
EXPECTED_FILES = {
    REFINE_DIR / "main.py": "939119d7e238058e1d200e6590cc73d3c6ad0ca8c88cbdab243cdd320571a2e3",
    REFINE_DIR / "utils.py": "c9da4b70568e847e081f6e2e74989499c7e3a13a76df60caf3c2dfc0e304c7b5",
    TEMPLATE: "7d6f45d1a599a0800a47cb6ae3ff056682801b50a634b7d36d0c8d78a4b0e609",
    MODEL / "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
}
FROZEN_EFFECTIVE_GEN = {
    "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "repetition_penalty": 1.05, "num_beams": 1,
    "eos_token_id": [151645, 151643], "pad_token_id": 151643, "bos_token_id": 151643,
    "max_new_tokens_call_arg": MAX_NEW_TOKENS,
}


def H(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append(p, rec):
    with open(p, "a") as f:
        f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "confirmatory"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--order", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--repeat-first-in-process", action="store_true",
                    help="smoke only: regenerate the first call_id again at the end, same process")
    ap.add_argument("--run-label", required=True)
    a = ap.parse_args()
    if a.repeat_first_in_process and a.mode != "smoke":
        raise SystemExit("--repeat-first-in-process is smoke-only")

    # ---- fail-closed environment binding ----
    commit = subprocess.check_output(["git", "-C", str(DIJA_ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert commit == EXPECTED_COMMIT, f"DIJA commit drift {commit}"
    for p, s in EXPECTED_FILES.items():
        assert sha_file(p) == s, f"SHA drift: {p}"
    assert sha_file(a.manifest) == a.manifest_sha256, "manifest SHA drift"
    rows = jl(a.manifest)
    assert rows and len({r["call_id"] for r in rows}) == len(rows), "duplicate call_ids"
    for r in rows:
        assert r["mode"] == a.mode, "manifest mode mismatch"
        assert SP.make_call_id(r["source_id"], r["branch"], r["candidate"], r["call_ordinal"]) == r["call_id"]
        assert SP.derive_seed(r["call_id"]) == r["seed"], f"seed mismatch {r['call_id']}"
        assert H(r["source_text"]) == r["source_text_sha256"]
    if a.mode == "confirmatory":
        assert all(r["candidate"] == 1 and r["call_ordinal"] == 1 for r in rows)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers
    det_runtime = SP.configure_deterministic_runtime()  # v2: before model load
    assert det_runtime == SP.DETERMINISTIC_RUNTIME, f"deterministic runtime drift: {det_runtime}"
    sys.path.insert(0, str(REFINE_DIR))
    from utils import Refiner  # M1 official DIJA refiner, SHA-verified above

    # Same constructor arguments M1's main.py passed via run (empty API fields -> local HF path)
    refiner = Refiner(hf_model_path=str(MODEL), api_model_name="", prompt_template_path=str(TEMPLATE),
                      attack_prompt="", output_json="", base_url="", api_key="")
    gc = refiner.model.generation_config
    eff = {"do_sample": gc.do_sample, "temperature": gc.temperature, "top_p": gc.top_p, "top_k": gc.top_k,
           "repetition_penalty": gc.repetition_penalty, "num_beams": gc.num_beams,
           "eos_token_id": list(gc.eos_token_id) if isinstance(gc.eos_token_id, (list, tuple)) else gc.eos_token_id,
           "pad_token_id": gc.pad_token_id, "bos_token_id": gc.bos_token_id,
           "max_new_tokens_call_arg": MAX_NEW_TOKENS}
    assert eff == FROZEN_EFFECTIVE_GEN, f"runtime generation config drift: {eff}"

    # pass-through capture of output ids (arguments and return value untouched)
    captured = {}
    _orig_generate = refiner.model.generate

    def _capture_generate(*args, **kwargs):
        out = _orig_generate(*args, **kwargs)
        captured["ids"] = out
        captured["input_len"] = kwargs["input_ids"].shape[1]
        return out
    refiner.model.generate = _capture_generate

    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "model_dtype": str(refiner.model.dtype), "host": os.uname().nodename,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "run_label": a.run_label,
           "seed_policy": "V2", "deterministic_runtime": det_runtime}
    print("ENV", json.dumps(env, sort_keys=True), flush=True)

    done = {x["call_id"] for x in jl(a.journal) if x.get("status") == "OK"}
    order = rows if a.order == "forward" else list(reversed(rows))
    plan = [(r, False) for r in order]
    if a.repeat_first_in_process:
        plan.append((order[0], True))

    for r, is_repeat in plan:
        if r["call_id"] in done and not is_repeat:
            continue
        prompt = refiner.apply_prompt_template(r["source_text"], refiner.template_str)
        seed = SP.derive_seed(r["call_id"])
        t0 = time.time()
        with torch.no_grad():
            SP.apply_seed(seed)  # immediately before the serial generate call
            response = refiner.qwen_generate(prompt, MAX_NEW_TOKENS)
        new_ids = captured["ids"][0][captured["input_len"]:].tolist()
        assert SP.read_deterministic_runtime() == SP.DETERMINISTIC_RUNTIME, "deterministic runtime changed mid-run"
        rec = {
            "status": "OK", "mode": a.mode, "run_label": a.run_label, "in_process_repeat": is_repeat,
            "call_id": r["call_id"], "source_id": r["source_id"], "branch": r["branch"],
            "candidate": r["candidate"], "call_ordinal": r["call_ordinal"], "seed": seed,
            "source_text_sha256": r["source_text_sha256"], "refinement_prompt_sha256": H(prompt),
            "Behavior": r["source_text"], "Refined_behavior": response,
            "response_sha256": H(response),
            "generated_token_ids_sha256": H(json.dumps(new_ids)), "n_generated_tokens": len(new_ids),
            "hit_max_new_tokens": len(new_ids) >= MAX_NEW_TOKENS,
            "seconds": round(time.time() - t0, 3), "env": env,
        }
        append(a.journal, rec)
        print(f"{r['call_id']} seed={seed} tokens={len(new_ids)} resp_sha={rec['response_sha256'][:16]}"
              f"{' (repeat)' if is_repeat else ''}", flush=True)
    print("DONE", a.run_label, flush=True)


if __name__ == "__main__":
    main()
