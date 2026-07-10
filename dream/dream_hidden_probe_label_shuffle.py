import json
import numpy as np

IN_JSON = "analysis_output/dream_hidden_probe_auc_torch.json"

rows = json.load(open(IN_JSON))

print("Loaded rows:", len(rows))

# This is only a reporting reminder:
# True label-shuffle needs retraining probes with shuffled y.
# Here we just tell you the next check to run from the torch script if needed.

print("\nTop AUC rows:")
for r in sorted(rows, key=lambda x: x["auc_mean"], reverse=True)[:20]:
    print(
        r["contrast"],
        r["scope"],
        "f=", r["frac"],
        "L", r["layer"],
        "AUC=", round(r["auc_mean"], 4),
        "n=", r["n"],
    )

print("\nIMPORTANT:")
print("AUC=1.0 is strong, but we should run a true shuffled-label probe next.")
