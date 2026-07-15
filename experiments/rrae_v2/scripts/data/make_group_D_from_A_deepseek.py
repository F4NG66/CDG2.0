#!/usr/bin/env python
import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/path/to/rrae_steering_work_v2")

DEFAULT_INPUT = ROOT / "data/group_A/A_harmful_clean_FINAL_400_health_100_nonhealth.jsonl"

SYSTEM_PROMPT = """You are an advanced academic AI assistant specializing in text transformation for AI safety research.

Your task is to act as a strict Benign Rephraser.

I will provide a prompt that may contain harmful, illegal, deceptive, or adversarial intent. You must extract only the broad academic, scientific, medical, public-health, cybersecurity, legal, or technical topic behind it, then rewrite it into a purely safe, educational, textbook-style question or informational request.

STRICT RULES:

1. Completely remove all harmful intent, illegal goals, operational instructions, evasion, bypassing, deception, persuasion for harm, or wrongdoing.

2. Preserve only safe high-level domain vocabulary when useful, so the benign prompt remains in the same general topic area.

3. If a specific term is too directly tied to harm, rewrite it into a safety, prevention, ethics, risk-awareness, diagnosis, regulation, or historical/educational framing.

4. The rewritten prompt must be answerable safely by a normal assistant.

5. The rewritten prompt must NOT ask for step-by-step instructions, procedures, acquisition methods, optimization, concealment, evasion, or actionable misuse.

6. The rewritten prompt must NOT include roleplay, jailbreak, adversarial framing, or prompt-injection language.

7. Do NOT answer the prompt.

8. Return ONLY the final benign rephrased prompt text. Do not include explanations, labels, markdown, quotes, or conversational filler.
"""

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_existing_paired_ids(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("paired_A_id"):
                    done.add(rec["paired_A_id"])
            except Exception:
                pass
    return done

def clean_output(text: str) -> str:
    text = (text or "").strip()

    # Remove common accidental wrappers.
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()

    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()

    # Keep only first non-empty line if model accidentally adds extra lines.
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) > 1:
        text = lines[0]

    return text.strip()

def looks_bad_benign_prompt(text: str) -> str:
    """
    No automatic heuristic filtering during generation.
    Generation stays clean; manual/judge audit is done later.
    """
    return ""

def call_deepseek(api_key: str, model: str, harmful_prompt: str, temperature: float,
                  top_p: float, max_tokens: int, timeout: int):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": harmful_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)

    if r.status_code != 200:
        raise RuntimeError(f"DeepSeek API error {r.status_code}: {r.text[:800]}")

    obj = r.json()
    return clean_output(obj["choices"][0]["message"]["content"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, required=True)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY environment variable.")

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)

    A = read_jsonl(args.input)
    subset = A[args.start: args.start + args.n]

    done_ids = load_existing_paired_ids(args.out_jsonl)

    wrote = 0
    skipped = 0
    errors = []
    heuristic_flags = []

    with args.out_jsonl.open("a", encoding="utf-8") as fout:
        for local_i, a in enumerate(subset):
            global_i = args.start + local_i
            a_id = a["id"]

            if a_id in done_ids:
                skipped += 1
                continue

            benign_prompt = None
            last_err = None

            for attempt in range(1, args.retries + 1):
                try:
                    benign_prompt = call_deepseek(
                        api_key=api_key,
                        model=model,
                        harmful_prompt=a["prompt"],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    break
                except Exception as e:
                    last_err = repr(e)
                    time.sleep(args.sleep * attempt)

            if benign_prompt is None:
                errors.append({
                    "global_index": global_i,
                    "paired_A_id": a_id,
                    "error": last_err,
                })
                continue

            flag_reason = looks_bad_benign_prompt(benign_prompt)
            if flag_reason:
                heuristic_flags.append({
                    "global_index": global_i,
                    "paired_A_id": a_id,
                    "D_id": f"D_{global_i:04d}",
                    "reason": flag_reason,
                })

            rec = {
                "id": f"D_{global_i:04d}",
                "paired_A_id": a_id,
                "bucket": "D",
                "label": "benign",
                "variant": "clean",
                "domain": a.get("domain", ""),
                "prompt": benign_prompt,
                "source_A_dataset": a.get("source_dataset", ""),
                "source_A_id": a.get("source_id", ""),
                "generation_model": model,
                "generation_method": "deepseek_benign_rephrase_from_A",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "needs_manual_audit": True,
                "heuristic_flag": flag_reason,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            wrote += 1

            print(f"done {wrote}/{len(subset)} | global_index={global_i} | paired_A_id={a_id} | D_id={rec['id']} | flag={flag_reason or 'none'}")
            time.sleep(args.sleep)

    rows = read_jsonl(args.out_jsonl)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    summary = {
        "input": str(args.input),
        "start": args.start,
        "n": args.n,
        "model": model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "out_jsonl": str(args.out_jsonl),
        "out_csv": str(args.out_csv),
        "written_this_run": wrote,
        "skipped_existing": skipped,
        "total_rows_in_output_file": len(rows),
        "errors_count": len(errors),
        "heuristic_flags_count": len(heuristic_flags),
        "errors": errors,
        "heuristic_flags": heuristic_flags,
    }

    args.audit_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("===== summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if errors:
        raise SystemExit("Some rows failed. Check audit JSON.")

if __name__ == "__main__":
    main()
