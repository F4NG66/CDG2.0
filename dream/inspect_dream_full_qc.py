import os
import json
import torch
from collections import Counter, defaultdict

ROOT = "outputs_dream_native_history_full"
MANIFEST = os.path.join(ROOT, "manifest.jsonl")

rows = [json.loads(x) for x in open(MANIFEST)]
print("manifest rows:", len(rows))

group_counts = Counter()
variant_counts = Counter()
bad_files = []
bad_vectors = []
expected_empty = 0
total_vectors = 0
response_lens = defaultdict(list)

for row in rows:
    path = row["path"]
    rec = torch.load(path, map_location="cpu")

    meta = rec.get("meta", {})
    case_id = meta.get("case_id", row.get("case_id", "NA"))
    variant = meta.get("variant", row.get("variant", "NA"))
    group = case_id[0]

    group_counts[group] += 1
    variant_counts[variant] += 1

    resp = rec.get("response_text", "")
    response_lens[group].append(len(resp.strip()))

    if "hidden" not in rec or "counts" not in rec:
        bad_files.append((path, "missing hidden/counts"))
        continue

    for scope, frac_map in rec["hidden"].items():
        for frac, layer_map in frac_map.items():
            count = rec["counts"].get(scope, {}).get(frac, None)

            for layer, vec in layer_map.items():
                total_vectors += 1
                v = vec.float()

                if count == 0:
                    expected_empty += 1
                    continue

                if not torch.isfinite(v).all():
                    bad_vectors.append((case_id, variant, scope, frac, layer, count))

print("\n===== COUNTS =====")
print("groups:", group_counts)
print("variants:", variant_counts)

print("\n===== VECTOR QC =====")
print("total_vectors:", total_vectors)
print("expected_empty_vectors:", expected_empty)
print("real_bad_vectors:", len(bad_vectors))
print("bad_files:", len(bad_files))

if bad_vectors[:20]:
    print("\nBAD VECTOR EXAMPLES:")
    for x in bad_vectors[:20]:
        print(x)

if bad_files[:20]:
    print("\nBAD FILE EXAMPLES:")
    for x in bad_files[:20]:
        print(x)

print("\n===== RESPONSE LENGTHS =====")
for g in sorted(response_lens):
    vals = response_lens[g]
    empty = sum(v == 0 for v in vals)
    mean_len = sum(vals) / len(vals)
    print(g, "n=", len(vals), "empty=", empty, "mean_len=", round(mean_len, 1), "min=", min(vals), "max=", max(vals))

print("\nQC_OK:", len(bad_files) == 0 and len(bad_vectors) == 0 and len(rows) == 400)
