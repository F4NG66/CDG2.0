import json

asr = json.load(open("analysis_output/dream_B_existing_deepseek_summary.json"))
cos = json.load(open("analysis_output/dream_direction_cosine.json"))
shuffle = json.load(open("analysis_output/dream_hidden_probe_label_shuffle_torch.json"))
auc = json.load(open("analysis_output/dream_hidden_probe_auc_torch.json"))

def best_auc(contrast):
    rows = [r for r in auc if r["contrast"] == contrast]
    return sorted(rows, key=lambda r: r["auc_mean"], reverse=True)[0]

def best_cos():
    return sorted(cos, key=lambda r: r["cos_BA_CD"], reverse=True)[0]

print("===== DREAM SUMMARY =====")
print(f"ASR: {asr['ASR']:.3f}")
print(f"Success count: {asr['success_count']} / {asr['n_judged_valid']}")
print(f"Labels: {asr['label_counts']}")

print("\n===== BEST AUC =====")
for c in [
    "B_vs_A_injection_harmful",
    "C_vs_D_injection_neutral",
    "B_vs_C_content_injected",
]:
    r = best_auc(c)
    print(
        f"{c}: AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f} "
        f"scope={r['scope']} frac={r['frac']} L{r['layer']} n={r['n']}"
    )

print("\n===== LABEL SHUFFLE =====")
for r in shuffle:
    print(
        f"{r['contrast']} {r['scope']} f={r['frac']} L{r['layer']}: "
        f"shuffle_AUC={r['shuffle_auc_mean']:.3f}±{r['shuffle_auc_std']:.3f}"
    )

print("\n===== SHARED DIRECTION =====")
r = best_cos()
print(
    f"best cos(BA,CD)={r['cos_BA_CD']:.3f} "
    f"scope={r['scope']} frac={r['frac']} L{r['layer']}"
)
print(f"cos(BA,BC)={r['cos_BA_BC']:.3f}")
print(f"cos(CD,BC)={r['cos_CD_BC']:.3f}")
