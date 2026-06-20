#!/usr/bin/env python3
"""p9_diagnose.py — SAE Analysis Diagnostics + Corrected Injection-Mechanism Probe Sweep.

Background
----------
The prior phase-3/4 analysis used B-vs-C as a proxy for "injection mechanism",
but B and C both have injection templates — the only variable is content danger.
This script corrects that by running the true injection contrasts:

  B-vs-A : harmful-injected vs harmful-clean  (content controlled, template varies)
  C-vs-D : neutral-injected vs neutral-clean  (content controlled, template varies)

Sub-tasks (--task)
------------------
  tfidf     — TF-IDF baseline classifier on raw behavior + response text for the
               three key pairs. Establishes the ceiling (B-vs-C) and floor
               (B-vs-A, C-vs-D) of surface-lexical separability. If the existing
               SAE probe AUC ≈ TF-IDF AUC for B-vs-C → SAE adds no value there.
  cv_check  — Validate that StratifiedKFold used in p3 is case-safe: check for
               multi-seed duplicates that would cause cross-fold leakage.
  probe_inj — 72-point probe sweep (3 scopes × 3 layers × 6 fracs × 2 spaces) for
               BOTH B-vs-A and C-vs-D using existing Phase-1 records + k=80 SAE.
               Scopes restricted to those present for all 4 groups:
               harm, out_mask, out_unmask.
  k_diag    — Re-encode recorded hidden states at k=160 / k=320 using existing SAE
               weights (TRAINER_TO_L0 maps trainer_1→k=80, _2→160, _3→320).
               Reports dead-feature fraction, reconstruction cosine-sim, and probe
               AUC on B-vs-A / C-vs-D. Auto-triggered when probe_inj AUC < --auc-thresh.
  all       — tfidf → cv_check → probe_inj, then k_diag if AUC is weak.

Limit parameter
---------------
  --limit N  Per-group record cap for probe_inj and k_diag (0 = all, default 30
             for fast comparison runs; set 0 for the full first run).
             TF-IDF and cv_check use all records (fast, no GPU needed).

Outputs
-------
  analysis_output/p9_results.json       — consolidated results
  analysis_output/p9_probe_inj_sweep.json — flat probe rows for p10 to consume
  analysis_output/p9_probe_inj_heatmap_{hidden,sae}.png

Usage
-----
  # Full first run (all records):
  python scripts/p9_diagnose.py --limit 0 --task all

  # Fast comparison run (30 per group):
  python scripts/p9_diagnose.py --limit 30 --task probe_inj

  # Force k diagnostic:
  python scripts/p9_diagnose.py --limit 30 --task k_diag --force-k-diag
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                             description=__doc__)
ap.add_argument("--out-dir",      default="outputs",
                help="Directory containing manifest.jsonl and .pt records")
ap.add_argument("--limit",        type=int, default=30,
                help="Per-group record cap for probe_inj/k_diag. 0=all.")
ap.add_argument("--task",         default="all",
                choices=["tfidf", "cv_check", "probe_inj", "k_diag", "all"],
                help="Which diagnostic sub-task(s) to run")
ap.add_argument("--auc-thresh",   type=float, default=0.65,
                help="B-vs-A/C-vs-D AUC below this auto-triggers k_diag")
ap.add_argument("--force-k-diag", action="store_true",
                help="Run k_diag regardless of probe_inj AUC")
ap.add_argument("--k-values",     nargs="+", type=int, default=[160, 320],
                help="k values to test in k_diag (alongside baseline k=80)")
ap.add_argument("--k-diag-layer", type=int, default=16,
                help="Layer to use for k_diag re-encoding")
ap.add_argument("--k-diag-scope", default="out_unmask",
                help="Scope to use for k_diag hidden state extraction")
ap.add_argument("--k-diag-frac",  type=float, default=0.10,
                help="Denoising fraction for k_diag")
ap.add_argument("--layers",       nargs="+", type=int, default=[11, 16, 26])
ap.add_argument("--fracs",        nargs="+", type=float,
                default=[0.05, 0.10, 0.20, 0.35, 0.50, 1.00])
ap.add_argument("--scopes",       nargs="+",
                default=["harm", "out_mask", "out_unmask"],
                help="Scopes to probe (must exist for all 4 groups)")
ap.add_argument("--no-plot",      action="store_true")
ap.add_argument("--sae-root",     default="./saes")
ap.add_argument("--out-json",     default="analysis_output/p9_results.json")
args = ap.parse_args()

os.makedirs("analysis_output", exist_ok=True)
results: dict = {}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _limit_records(records: list, limit: int, groups=("A", "B", "C", "D")) -> list:
    """Return records keeping at most `limit` per group. 0 = keep all."""
    if limit <= 0:
        return records
    from cdg.probe import group_letter
    counts: dict = {}
    out = []
    for r in records:
        g = group_letter(r)
        if g not in groups:
            out.append(r)
            continue
        counts.setdefault(g, 0)
        if counts[g] < limit:
            out.append(r)
            counts[g] += 1
    return out


def _print_section(title: str):
    print(f"\n{'━' * 62}")
    print(f"  {title}")
    print('━' * 62)


def _group_counts(records) -> dict:
    from cdg.probe import group_letter
    counts: dict = {}
    for r in records:
        g = group_letter(r)
        counts[g] = counts.get(g, 0) + 1
    return counts


def _counts_str(records) -> str:
    return "  ".join(f"{g}={n}" for g, n in sorted(_group_counts(records).items()))


# ─── TASK A: TF-IDF Baseline ──────────────────────────────────────────────────

def run_tfidf() -> dict:
    _print_section("Task A — TF-IDF Baseline (surface-lexical separability)")

    from cdg.data import load_cdg_root
    from cdg.probe import load_records

    cases = load_cdg_root("./prompts/cdg_injection")
    all_records = load_records(args.out_dir)
    resp_map = {r.get("case_id"): r.get("response_text", "") for r in all_records}

    # -- helper: extract texts + binary labels for a pair of group letters -----
    def _texts_labels(g_pos: str, g_neg: str, field: str):
        texts, labels = [], []
        for c in cases:
            if c.group_letter not in (g_pos, g_neg):
                continue
            if field == "behavior":
                txt = c.behavior
            elif field == "response":
                txt = resp_map.get(c.case_id, "")
            else:  # combined
                txt = c.behavior + " " + resp_map.get(c.case_id, "")
            texts.append(txt)
            labels.append(1 if c.group_letter == g_pos else 0)
        return texts, np.array(labels)

    tfidf_rows = {}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score

        pairs = [
            ("B", "C", "B-vs-C (content danger; same template structure)"),
            ("B", "A", "B-vs-A (injection mechanism; same harmful content)"),
            ("C", "D", "C-vs-D (injection mechanism; same neutral content)"),
        ]

        print(f"\n{'Pair':<8} {'Description':<46} {'field':<10} {'AUC':>7}  {'AUC±':>6}")
        print("─" * 84)

        for g_pos, g_neg, desc in pairs:
            key = f"{g_pos}_vs_{g_neg}"
            tfidf_rows[key] = {"description": desc}

            for field in ["behavior", "response", "combined"]:
                texts, labels = _texts_labels(g_pos, g_neg, field)
                if len(set(labels.tolist())) < 2 or len(texts) < 4:
                    tfidf_rows[key][field] = {"auc": float("nan"), "n": len(texts)}
                    print(f"{key:<8} {desc:<46} {field:<10} {'n/a':>7}")
                    continue

                pipe = make_pipeline(
                    TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                    sublinear_tf=True),
                    LogisticRegression(C=1.0, max_iter=500))

                k = min(5, int(np.bincount(labels).min()))
                if k < 2:
                    tfidf_rows[key][field] = {"auc": float("nan"), "n": len(texts),
                                              "note": "class too small"}
                    continue

                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
                aucs = []
                for tr, te in skf.split(texts, labels):
                    X_tr = [texts[i] for i in tr]
                    X_te = [texts[i] for i in te]
                    try:
                        pipe.fit(X_tr, labels[tr])
                        probs = pipe.predict_proba(X_te)[:, 1]
                        aucs.append(roc_auc_score(labels[te], probs))
                    except ValueError:
                        pass

                auc     = float(np.mean(aucs)) if aucs else float("nan")
                auc_std = float(np.std(aucs))  if aucs else float("nan")
                tfidf_rows[key][field] = {"auc": auc, "auc_std": auc_std, "n": len(texts)}
                print(f"{key:<8} {desc:<46} {field:<10} {auc:7.3f}  {auc_std:6.3f}")

        # -- load existing SAE B-vs-C AUC for comparison ----------------------
        sae_bc_auc = float("nan")
        probe_json = "analysis_output/probe_sweep.json"
        if os.path.exists(probe_json):
            with open(probe_json) as fh:
                probe_rows = json.load(fh)
            best = max(
                (r for r in probe_rows
                 if r.get("auc") == r.get("auc") and r.get("scope") == "tpl_mask"),
                key=lambda r: r.get("auc", 0), default=None)
            if best:
                sae_bc_auc = best.get("auc", float("nan"))

        # -- interpretation ---------------------------------------------------
        print("\nKey observations:")
        bc_beh = (tfidf_rows.get("B_vs_C", {}).get("behavior") or {}).get("auc", float("nan"))
        ba_beh = (tfidf_rows.get("B_vs_A", {}).get("behavior") or {}).get("auc", float("nan"))
        cd_beh = (tfidf_rows.get("C_vs_D", {}).get("behavior") or {}).get("auc", float("nan"))

        if bc_beh == bc_beh and bc_beh > 0.85:
            print(f"  • B-vs-C behavior TF-IDF AUC = {bc_beh:.3f}  (SAE p3 best = {sae_bc_auc:.3f})")
            print(f"    → B-vs-C is trivially separable by behavior text alone.")
            print(f"      The near-perfect SAE probe AUC reflects CONTENT DANGER, not injection.")
        if ba_beh == ba_beh and ba_beh < 0.60:
            print(f"  • B-vs-A behavior TF-IDF AUC = {ba_beh:.3f}  (same behavior, different template)")
            print(f"    → Any SAE probe AUC > {ba_beh:.3f} on B-vs-A signals genuine injection detection.")
        if cd_beh == cd_beh and cd_beh < 0.60:
            print(f"  • C-vs-D behavior TF-IDF AUC = {cd_beh:.3f}  (same behavior, different template)")
            print(f"    → Any SAE probe AUC > {cd_beh:.3f} on C-vs-D signals genuine injection detection.")

        tfidf_rows["sae_bc_reference"] = {"best_auc": sae_bc_auc,
                                           "scope": "tpl_mask", "source": "probe_sweep.json"}

    except ImportError as e:
        print(f"[skip] sklearn not available: {e}")
        tfidf_rows["error"] = str(e)

    results["tfidf"] = tfidf_rows
    return tfidf_rows


# ─── TASK B: CV Grouping Check ────────────────────────────────────────────────

def run_cv_check() -> dict:
    _print_section("Task B — CV Grouping Check (case-safe fold splitting)")

    from cdg.probe import load_records, group_letter
    from collections import defaultdict

    records = load_records(args.out_dir)
    case_entries: dict = defaultdict(list)
    for r in records:
        case_entries[r.get("case_id", "")].append({
            "seed": r.get("seed", 0),
            "group": group_letter(r),
        })

    group_counts = _group_counts(records)
    multi_seed = {cid: v for cid, v in case_entries.items() if len(v) > 1}

    print(f"\nTotal records : {len(records)}")
    print(f"Group sizes   : {_counts_str(records)}")
    print(f"Unique case_ids : {len(case_entries)}")
    print(f"Multi-record case_ids : {len(multi_seed)}")

    verdict = "OK"
    recommendation = "StratifiedKFold is case-safe (one record per case)."

    if multi_seed:
        verdict = "NEEDS_GROUP_KFOLD"
        recommendation = (
            "Same case_id appears in >1 record. "
            "Use GroupKFold(groups=case_id) to prevent cross-fold leakage."
        )
        print(f"\n  ⚠  {len(multi_seed)} case_id(s) appear more than once:")
        for cid, entries in list(multi_seed.items())[:5]:
            print(f"     {cid}: {entries}")
        if len(multi_seed) > 5:
            print(f"     ... ({len(multi_seed) - 5} more)")
        print(f"\n  → {recommendation}")
    else:
        print(f"\n  ✓  Each case_id is unique — {recommendation}")

    cv_result = {
        "total_records": len(records),
        "group_counts": group_counts,
        "unique_case_ids": len(case_entries),
        "multi_record_cases": len(multi_seed),
        "verdict": verdict,
        "recommendation": recommendation,
    }
    results["cv_check"] = cv_result
    return cv_result


# ─── TASK C: Injection-Mechanism Probe Sweep (B-vs-A, C-vs-D) ────────────────

def run_probe_inj() -> dict:
    _print_section("Task C — Injection-Mechanism Probe Sweep (B-vs-A, C-vs-D)")

    from cdg.probe import load_records, probe_sweep, group_letter

    all_records = load_records(args.out_dir)
    records = _limit_records(all_records, args.limit)

    print(f"limit={args.limit} → records: {_counts_str(records)}")
    print(f"scopes: {args.scopes}")
    print(f"layers: {args.layers}  fracs: {[f'{f:.2f}' for f in args.fracs]}")
    n_grid = len(args.scopes) * len(args.layers) * len(args.fracs)
    print(f"Grid points per pair per space: {n_grid}")

    pairs = [
        {"pos": ("B",), "neg": ("A",), "label": "B_vs_A",
         "desc": "harmful injected vs harmful clean (injection mechanism)"},
        {"pos": ("C",), "neg": ("D",), "label": "C_vs_D",
         "desc": "neutral injected vs neutral clean (injection mechanism)"},
    ]

    all_inj_results: dict = {}

    for space in ["hidden", "sae"]:
        for pair in pairs:
            key = f"{pair['label']}_{space}"
            print(f"\n── {pair['label']}  space={space}  ({pair['desc']}) ──")

            rows = probe_sweep(
                records,
                scopes=args.scopes,
                fracs=args.fracs,
                layers=args.layers,
                space=space,
                pos_groups=pair["pos"],
                neg_groups=pair["neg"],
            )
            for r in rows:
                r["space"] = space
                r["pair"] = pair["label"]

            valid = [r for r in rows
                     if r.get("auc") == r.get("auc") and r.get("n", 0) > 0]

            if not valid:
                print("  [!] No valid results — check that these scopes have "
                      "data for both groups in this pair.")
                all_inj_results[key] = rows
                continue

            rows_sorted = sorted(valid, key=lambda r: -(r.get("auc") or 0))
            print(f"\n  {'scope':12} {'frac':6} {'layer':6} {'AUC':8} {'F1':8} "
                  f"{'n':5}  {'AUC_std':8}")
            print("  " + "─" * 62)
            for r in rows_sorted:
                auc = r.get("auc", float("nan"))
                f1  = r.get("macro_f1", float("nan"))
                std = r.get("auc_std", float("nan"))
                print(f"  {r['scope']:12} {r['frac']:6.2f} {r['layer']:6d} "
                      f"{auc:8.3f} {f1:8.3f} {r.get('n', 0):5d}  {std:8.3f}")

            best = rows_sorted[0]
            print(f"\n  >> Best: scope={best['scope']}  frac={best['frac']:.2f}  "
                  f"layer={best['layer']}  AUC={best.get('auc', float('nan')):.3f}  "
                  f"AUC_std={best.get('auc_std', float('nan')):.3f}")
            all_inj_results[key] = rows

    # -- interpretation -------------------------------------------------------
    print("\nSignal assessment:")
    for space in ["hidden", "sae"]:
        for pair in pairs:
            key = f"{pair['label']}_{space}"
            rows = all_inj_results.get(key, [])
            valid = [r for r in rows
                     if r.get("auc") == r.get("auc") and r.get("n", 0) > 0]
            if not valid:
                continue
            best_auc = max(r.get("auc", 0) for r in valid)
            if best_auc >= 0.80:
                symbol = "✓"
                msg = f"strong signal — SAE captures injection template footprint"
            elif best_auc >= args.auc_thresh:
                symbol = "⚠"
                msg = f"moderate signal — consider k_diag"
            else:
                symbol = "✗"
                msg = f"weak signal (< {args.auc_thresh}) — suggest running k_diag"
            print(f"  {symbol}  {pair['label']} ({space:6}): best AUC={best_auc:.3f}  {msg}")

    results["probe_inj"] = all_inj_results

    # -- plot -----------------------------------------------------------------
    if not args.no_plot:
        _plot_probe_inj(all_inj_results)

    return all_inj_results


def _plot_probe_inj(all_results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pairs  = ["B_vs_A", "C_vs_D"]
        spaces = ["hidden", "sae"]
        fracs  = sorted(args.fracs)
        layers = sorted(args.layers)
        scopes = args.scopes

        for space in spaces:
            fig, axes = plt.subplots(
                len(pairs), len(layers),
                figsize=(5 * len(layers), 4.5 * len(pairs)),
                squeeze=False)

            for pi, pair in enumerate(pairs):
                rows = all_results.get(f"{pair}_{space}", [])
                for li, layer in enumerate(layers):
                    ax = axes[pi][li]
                    scopes_with_data = [
                        s for s in scopes
                        if any(r["scope"] == s and r.get("n", 0) > 0
                               and r.get("auc") == r.get("auc")
                               for r in rows)]
                    if not scopes_with_data:
                        ax.set_title(f"{pair}  L{layer}\n(no data)", fontsize=8)
                        ax.axis("off")
                        continue

                    mat = np.full((len(scopes_with_data), len(fracs)), float("nan"))
                    for si, scope in enumerate(scopes_with_data):
                        for fi, frac in enumerate(fracs):
                            hit = next(
                                (r for r in rows
                                 if r["scope"] == scope and r["layer"] == layer
                                 and abs(r.get("frac", -1) - frac) < 1e-4
                                 and r.get("n", 0) > 0), None)
                            if hit and hit.get("auc") == hit.get("auc"):
                                mat[si, fi] = hit["auc"]

                    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.4, vmax=1.0,
                                   aspect="auto", interpolation="nearest")
                    ax.set_xticks(range(len(fracs)))
                    ax.set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=7)
                    ax.set_yticks(range(len(scopes_with_data)))
                    ax.set_yticklabels(scopes_with_data, fontsize=7)
                    ax.set_xlabel("Denoising fraction", fontsize=8)
                    ax.set_title(f"{pair}  L{layer}", fontsize=9)

                    for si in range(len(scopes_with_data)):
                        for fi in range(len(fracs)):
                            v = mat[si, fi]
                            if not np.isnan(v):
                                ax.text(fi, si, f"{v:.2f}", ha="center", va="center",
                                        fontsize=6,
                                        color="white" if v > 0.8 else "black")

                plt.colorbar(im, ax=axes[pi], label=f"AUC ({pair})", fraction=0.02)

            plt.suptitle(
                f"Injection-Mechanism Probe Sweep  (space={space})\n"
                f"B-vs-A: harmful injected vs clean | C-vs-D: neutral injected vs clean",
                fontsize=9)
            plt.tight_layout()
            path = f"analysis_output/p9_probe_inj_heatmap_{space}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {path}")

    except ImportError as e:
        print(f"[skip plot] {e}")


# ─── TASK D: k-SAE Diagnostic ─────────────────────────────────────────────────

def run_k_diag() -> dict:
    _print_section("Task D — k-SAE Diagnostic (k=80 vs larger k, same weights)")

    from cdg.probe import load_records, stack_group, linear_probe
    from cdg.sae import load_sae, sae_ckpt_path
    from cdg.config import TRAINER_TO_L0, L0_TO_TRAINER

    all_records = load_records(args.out_dir)
    records = _limit_records(all_records, args.limit)
    print(f"limit={args.limit} → records: {_counts_str(records)}")

    layer = args.k_diag_layer
    scope = args.k_diag_scope
    frac  = args.k_diag_frac
    print(f"Diagnostic at: scope={scope}  frac={frac}  layer={layer}")
    print(f"Note: using same SAE weights (trainer_1, k=80) with k overridden at inference.")
    print(f"      This tests whether the encoder has learned features beyond the k=80 cutoff.")

    # Find SAE checkpoint (trainer_1 = k=80 baseline)
    ckpt, cfgp = sae_ckpt_path(args.sae_root + "/llada_mask", layer, trainer=1)
    if not os.path.exists(ckpt):
        msg = f"SAE checkpoint not found: {ckpt}"
        print(f"[error] {msg}")
        results["k_diag"] = {"error": msg}
        return {}

    # Load hidden states once
    hidden_by_group: dict = {}
    for grp in ("A", "B", "C", "D"):
        X_h, _ = stack_group(records, groups=(grp,), scope=scope,
                              frac=frac, layer=layer, space="hidden")
        if X_h is not None:
            hidden_by_group[grp] = X_h.float()

    if not hidden_by_group:
        msg = f"No hidden states found at scope={scope} frac={frac} layer={layer}"
        print(f"[error] {msg}")
        results["k_diag"] = {"error": msg}
        return {}

    X_all = torch.cat(list(hidden_by_group.values()), 0)
    print(f"Hidden-state matrix: {X_all.shape}  (n_samples × d_model)")

    k_results: dict = {}
    k_values = sorted(set([80] + list(args.k_values)))

    print(f"\n{'k':>5}  {'live%':>7}  {'dead%':>7}  {'meanL0':>7}  "
          f"{'cos_sim':>8}  {'B-vs-A AUC':>11}  {'C-vs-D AUC':>11}")
    print("─" * 72)

    for k_val in k_values:
        sae = load_sae(ckpt, config_path=cfgp, k=k_val)
        sae.eval()

        # Dead-feature analysis over all hidden states
        z_all = sae.encode(X_all)                          # (N, n_features)
        n_active  = int((z_all > 0).any(0).sum().item())
        n_dead    = sae.n_features - n_active
        dead_frac = n_dead / sae.n_features
        mean_l0   = float((z_all > 0).sum(1).float().mean().item())

        # Reconstruction quality
        x_hat   = sae.decode(z_all)
        cos_sim = float(F.cosine_similarity(X_all, x_hat, dim=-1).mean().item())
        mse     = float(F.mse_loss(x_hat, X_all).item())

        # Probe AUC on B-vs-A and C-vs-D using re-encoded features
        probe_aucs: dict = {}
        for pos_grp, neg_grp in [("B", "A"), ("C", "D")]:
            if pos_grp not in hidden_by_group or neg_grp not in hidden_by_group:
                probe_aucs[f"{pos_grp}vs{neg_grp}"] = float("nan")
                continue
            z_pos = sae.encode(hidden_by_group[pos_grp]).numpy()
            z_neg = sae.encode(hidden_by_group[neg_grp]).numpy()
            X_sae = np.concatenate([z_pos, z_neg], 0)
            y_sae = np.concatenate([np.ones(len(z_pos)), np.zeros(len(z_neg))]).astype(int)
            m = linear_probe(X_sae, y_sae)
            probe_aucs[f"{pos_grp}vs{neg_grp}"] = m.get("auc", float("nan"))

        ba_auc = probe_aucs.get("BvsA", float("nan"))
        cd_auc = probe_aucs.get("CvsD", float("nan"))

        print(f"{k_val:>5}  {100*(1-dead_frac):>7.1f}  {100*dead_frac:>7.1f}  "
              f"{mean_l0:>7.1f}  {cos_sim:>8.4f}  {ba_auc:>11.3f}  {cd_auc:>11.3f}")

        k_results[k_val] = {
            "k": k_val, "n_features": sae.n_features,
            "n_active": n_active, "n_dead": n_dead,
            "dead_frac": float(dead_frac), "mean_l0": float(mean_l0),
            "recon_cosine_sim": float(cos_sim), "recon_mse": float(mse),
            "probe_aucs": probe_aucs,
        }

    # -- verdict --------------------------------------------------------------
    if len(k_results) >= 2:
        k80  = k_results.get(80, {})
        k_larger = {kv: k_results[kv] for kv in k_results if kv > 80}

        def _auc_improvement(base_r, new_r, pair):
            base = base_r.get("probe_aucs", {}).get(pair, float("nan"))
            new  = new_r.get("probe_aucs", {}).get(pair, float("nan"))
            if base != base or new != new:
                return 0.0
            return new - base

        dead_improved = any(
            (v.get("dead_frac", 1.0) < k80.get("dead_frac", 1.0) - 0.05)
            for v in k_larger.values())
        auc_improved = any(
            _auc_improvement(k80, v, "BvsA") > 0.05 or
            _auc_improvement(k80, v, "CvsD") > 0.05
            for v in k_larger.values())

        print("\nVerdict:")
        if dead_improved and auc_improved:
            print("  → BOTTLENECK: k=80 is too restrictive.")
            print("    Dead features decrease AND probe AUC improves with larger k.")
            print("    Recommend: switch to larger k for steps 5-7.")
            verdict = "k_is_bottleneck"
        elif dead_improved and not auc_improved:
            print("  → UTILISATION ISSUE ONLY: more features activate with larger k,")
            print("    but probe AUC on injection pairs does NOT improve.")
            print("    k=80 restricts overall utilisation but is NOT the root cause")
            print("    of weak injection signal. Continue with k=80 for Steps 5-7.")
            print("    (Note the utilisation limitation in the paper write-up.)")
            verdict = "dead_only_not_probe"
        else:
            print("  → k IS NOT THE BOTTLENECK: neither dead-feature rate nor probe AUC")
            print("    changes substantially. The encoder does not have more useful features")
            print("    above the k=80 threshold for this injection detection task.")
            print("    Consider: raw hidden-state analysis without SAE (Step 5-7 via p10).")
            verdict = "k_not_bottleneck"

        k_results["_verdict"] = verdict

    results["k_diag"] = k_results
    return k_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    task = args.task

    if task in ("tfidf", "all"):
        run_tfidf()

    if task in ("cv_check", "all"):
        run_cv_check()

    probe_inj_results: dict = {}
    if task in ("probe_inj", "all"):
        probe_inj_results = run_probe_inj()

    # Auto-trigger k_diag if injection signal is weak
    trigger_k = args.force_k_diag or task == "k_diag"
    if task == "all" and not trigger_k and probe_inj_results:
        for key, rows in probe_inj_results.items():
            valid = [r for r in rows
                     if r.get("auc") == r.get("auc") and r.get("n", 0) > 0]
            if valid:
                best_auc = max(r.get("auc", 0) for r in valid)
                if best_auc < args.auc_thresh:
                    print(f"\n  [auto] Triggering k_diag: {key} best AUC={best_auc:.3f} "
                          f"< threshold={args.auc_thresh}")
                    trigger_k = True
                    break

    if trigger_k:
        run_k_diag()

    # -- Save consolidated results --------------------------------------------
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n[saved] {args.out_json}")

    # Flat probe_inj rows for p10 to read
    if "probe_inj" in results:
        inj_path = "analysis_output/p9_probe_inj_sweep.json"
        flat: list = []
        for rows in results["probe_inj"].values():
            flat.extend(rows if isinstance(rows, list) else [])
        with open(inj_path, "w") as fh:
            json.dump(flat, fh, indent=2, default=float)
        print(f"[saved] {inj_path}  (flat rows for p10)")


if __name__ == "__main__":
    main()
