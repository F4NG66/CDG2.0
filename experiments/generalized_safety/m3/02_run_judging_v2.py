"""
M3 BUILD+DEV baseline judging — executor, VERSION 2.

Identical judge runtime/contract to v1's 02_run_judging.py (deepseek-v4-pro,
temperature 0, thinking disabled, JSON-object response format, frozen
4-attempt retry, returned-model drift check). Points at the v2 packet (base
requests corrected for RENeLLM). The 800 DIJA judgments were pre-seeded into
DEEPSEEK_JUDGMENTS_BLINDED_1528.jsonl verbatim from v1 (base_request/
model_response unchanged for DIJA, so re-judging would be pure API spend with
no possible different answer) — this script, being resumable exactly like v1,
skips every blinded_id already present and calls the API only for the 728
corrected RENeLLM rows.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"
MAX_ATTEMPTS = 4
PACKET = HERE / "JUDGE_PACKET_BLINDED_1528.jsonl"
PACKET_SHA = "cae048433789900993d24d8cbb167a3aedceae745f926cfd01ecc65b00c833d6"
SYSTEM_PROMPT_PATH = HERE / "JUDGE_SYSTEM_PROMPT.txt"
RUNTIME = HERE / "judging_execution"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def api_json(url, api_key, payload, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def validate(row, parsed):
    require(isinstance(parsed, dict), "Non-dict judge payload")
    label = parsed.get("outcome_label")
    conf = parsed.get("reviewer_confidence")
    notes = parsed.get("reviewer_notes")
    require(label in row["allowed_labels"], f"Label outside allowed set: {label!r}")
    require(conf in row["allowed_confidence"], f"Invalid confidence: {conf!r}")
    require(isinstance(notes, str) and notes.strip(), "Empty reviewer notes")
    return label, conf, notes.strip()


def judge_one(row, api_key, journal):
    user_prompt = (
        f"request_type: {row['request_type']}\n"
        f"domain: {row['domain']}\n\n"
        f"base_request:\n{row['base_request']}\n\n"
        f"model_response:\n{row['model_response']}\n\n"
        f"allowed_outcome_labels:\n"
        f"{json.dumps(row['allowed_labels'], ensure_ascii=False)}"
    )
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}],
               "temperature": 0, "max_tokens": 160,
               "response_format": {"type": "json_object"},
               "thinking": {"type": "disabled"}}
    errors = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            body = api_json(API_URL, api_key, payload)
            returned = str(body.get("model", ""))
            require(returned == MODEL, f"Returned model drift: {returned!r}")
            content = body["choices"][0]["message"]["content"]
            require(isinstance(content, str) and content.strip(), "Empty API content")
            parsed = json.loads(content)
            label, conf, notes = validate(row, parsed)
            journal.write(json.dumps({
                "blinded_id": row["blinded_id"], "attempt": attempt, "status": "OK",
                "raw_content": content, "model_returned": returned,
                "usage": body.get("usage") or {}}, sort_keys=True) + "\n")
            return {"blinded_id": row["blinded_id"],
                    "blinded_ordinal": row["blinded_ordinal"],
                    "request_type": row["request_type"],
                    "outcome_label": label, "reviewer_confidence": conf,
                    "reviewer_notes": notes,
                    "judge_model_requested": MODEL, "judge_model_returned": returned,
                    "temperature": 0, "thinking": "disabled", "attempts": attempt,
                    "raw_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "usage": body.get("usage") or {}}
        except Exception as e:                                    # noqa: BLE001
            errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
            journal.write(json.dumps({"blinded_id": row["blinded_id"],
                                      "attempt": attempt, "status": "ERROR",
                                      "error": str(e)[:400]}, sort_keys=True) + "\n")
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"judge failed for {row['blinded_id']}: {errors}")


def main():
    global SYSTEM_PROMPT
    require(sha_file(PACKET) == PACKET_SHA, "packet SHA mismatch — refusing to run")
    SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    require(api_key, "DEEPSEEK_API_KEY not set")

    RUNTIME.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in PACKET.open() if l.strip()]
    out_path = HERE / f"DEEPSEEK_JUDGMENTS_BLINDED_{len(rows)}.jsonl"

    done = {}
    if out_path.exists():
        for l in out_path.open():
            if l.strip():
                r = json.loads(l)
                done[r["blinded_id"]] = r
    print(f"{len(rows)} items; {len(done)} already judged (pre-seeded DIJA + any resumed RENeLLM)", flush=True)

    with (RUNTIME / "call_journal.jsonl").open("a") as journal, \
         out_path.open("a") as out:
        for i, row in enumerate(rows, 1):
            if row["blinded_id"] in done:
                continue
            rec = judge_one(row, api_key, journal)
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    print("JUDGING_COMPLETE", sha_file(out_path), flush=True)


if __name__ == "__main__":
    main()
