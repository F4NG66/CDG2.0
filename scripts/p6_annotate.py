#!/usr/bin/env python3
"""p6_annotate.py — Cluster Annotation + Activation Traces.

Merges s11_annotate.py (cluster-level LLM annotation) and
s6_activation_trace.py (per-module activation traces over denoising).

Two-level annotation:
  Per-feature: logit-lens vocab labels (from vocab_labels.json cache or SAE weights)
  Per-cluster: aggregate top tokens → LLM one-sentence theme

Per-module activation traces:
  For each module's top features, plot activation across denoising fracs
  for groups B (harmful injection) vs C (neutral injection) vs A (harmful clean).
  Shows WHEN the model "commits" to harmful content within each feature program.

Usage
-----
python scripts/p6_annotate.py outputs_new2/ \\
    --module-csv outputs/analysis/modules/feature_module.csv \\
    --vocab-labels-json analysis_output/vocab_labels.json \\
    --records-dir outputs/ \\
    --tokenizer-path /path/to/llada \\
    --api-model deepseek-chat

Outputs (under <out_dir>/analysis/annotation/)
-----------------------------------------------
  feature_labels.csv       per-feature vocab + actual activation tokens
  module_labels.csv        per-module theme + keywords
  module_annotation.png    bar chart of module sizes + theme table
  traces/module_{id}.png   activation trace per module (B vs C vs A)
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_SKIP_TOKENS = {
    "<|endoftext|>", "<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>",
    "\n", "\r\n", "\r", " ", "", "\t", "eter",
}


# ---------------------------------------------------------------------------
# Vocab label loading
# ---------------------------------------------------------------------------

def load_cached_vocab(json_path: str) -> dict[int, dict]:
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    result = {}
    if isinstance(raw, list):
        for e in raw:
            if isinstance(e, dict) and "feature" in e:
                result[int(e["feature"])] = e
    elif isinstance(raw, dict):
        for k, v in raw.items():
            try: result[int(k)] = v
            except (ValueError, TypeError): pass
    return result


def aggregate_cluster_tokens(feature_ids, vocab_labels, actual_top_tokens,
                              topk=10, min_support=2):
    top_c, bot_c, act_c = Counter(), Counter(), Counter()
    for fid in feature_ids:
        vl = vocab_labels.get(int(fid), {})
        for t in (vl.get("top_tokens") or [])[:topk]:
            s = t.get("token_str", "").strip()
            if s and s not in _SKIP_TOKENS: top_c[s] += 1
        for t in (vl.get("bottom_tokens") or [])[:topk]:
            s = t.get("token_str", "").strip()
            if s and s not in _SKIP_TOKENS: bot_c[s] += 1
        for s in actual_top_tokens.get(int(fid), []):
            if s and s not in _SKIP_TOKENS: act_c[s] += 1

    def rank(c):
        return [(t, n) for t, n in c.most_common(25) if n >= min_support]

    return {"top_union": rank(top_c), "bottom_union": rank(bot_c),
            "actual_union": rank(act_c)}


# ---------------------------------------------------------------------------
# LLM annotation
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are an interpretability researcher studying a Chinese diffusion language model SAE. "
    "Given a group of co-activating SAE features (a cluster/module), identify their shared theme. "
    "[LOGIT-LENS] tokens are vocabulary items each feature geometrically promotes. "
    "[ACTUAL] tokens are those that activated features in real model outputs. "
    "[METADATA] lines describe experimental conditions — NOT part of model text. "
    "Write ONE precise sentence for the cluster theme. Then list 3-5 keywords. "
    'JSON only: {"theme":"<sentence>","keywords":["k1","k2",...],"confidence":"high"|"medium"|"low"}'
)


class _Client:
    def __init__(self, api_key, base_url, model, timeout=60, retries=3):
        import requests as _r
        self._r = _r
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = api_key; self.model = model
        self.timeout = timeout; self.retries = retries

    def call(self, mid, n_feats, agg, feat_summaries):
        top = ", ".join(f"{t}(×{c})" for t, c in agg["top_union"][:15])
        bot = ", ".join(f"{t}(×{c})" for t, c in agg["bottom_union"][:8])
        act = ", ".join(f"{t}(×{c})" for t, c in agg["actual_union"][:15])
        fs  = "\n".join(f"  • {s}" for s in feat_summaries[:8])
        msg = (f"Module {mid} ({n_feats} features)\n\n"
               f"[LOGIT-LENS promotes] {top or '(none)'}\n"
               f"[LOGIT-LENS suppresses] {bot or '(none)'}\n"
               f"[ACTUAL high-activation tokens] {act or '(none)'}\n"
               f"[Per-feature vocab]\n{fs}\n\n"
               "What unifies this feature cluster?")
        payload = {"model": self.model, "temperature": 0.0,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user",   "content": msg}]}
        hdrs = {"Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json"}
        last = None
        for attempt in range(self.retries):
            try:
                r = self._r.post(self.url, json=payload, headers=hdrs,
                                 timeout=self.timeout)
                r.raise_for_status()
                import re
                c = r.json()["choices"][0]["message"]["content"]
                c = re.sub(r"^```(json)?|```$", "", c, flags=re.M).strip()
                d = json.loads(c)
                return {"theme": str(d.get("theme", "")),
                        "keywords": d.get("keywords", []),
                        "confidence": str(d.get("confidence", "medium"))}
            except Exception as e:
                last = e; time.sleep(1.5 * (attempt + 1))
        return {"theme": f"[api_error: {last}]", "keywords": [], "confidence": "low"}


# ---------------------------------------------------------------------------
# Activation traces (from s6_activation_trace.py)
# ---------------------------------------------------------------------------

def plot_module_traces(module_id, feature_ids_top, records_dir,
                       scope, layer, save_dir, groups=("B", "C", "A")):
    """Plot activation over denoising fracs for top features of a module."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cdg.probe import load_records, stack_group
        from cdg.config import get_backend_config
    except ImportError as e:
        warnings.warn(f"Cannot plot traces: {e}")
        return

    cfg = get_backend_config("llada_attack")
    records = load_records(records_dir, cfg.name)

    # Discover fracs
    all_fracs = []
    for r in records:
        rec = r.get("_rec") or {}
        store = rec.get("sae", {}).get(scope, {})
        if store:
            all_fracs = sorted(float(k) for k in store.keys())
            break
    if not all_fracs:
        return

    colors = {"B": "#e15759", "C": "#4e79a7", "A": "#59a14f", "D": "#f28e2b"}
    linestyles = {"B": "-", "C": "--", "A": ":", "D": "-."}

    fig, axes = plt.subplots(1, len(feature_ids_top),
                              figsize=(4.5 * len(feature_ids_top), 4),
                              squeeze=False)

    for fi, fid in enumerate(feature_ids_top):
        ax = axes[0][fi]
        for grp in groups:
            grp_acts = []
            for frac in all_fracs:
                X, _ = stack_group(records, groups=(grp,), scope=scope,
                                   frac=frac, layer=layer, space="sae")
                val = float(X[:, fid].mean()) if X is not None else float("nan")
                grp_acts.append(val)
            if not all(np.isnan(grp_acts)):
                ax.plot(all_fracs, grp_acts,
                        color=colors.get(grp, "gray"),
                        linestyle=linestyles.get(grp, "-"),
                        linewidth=2, marker="o", markersize=4,
                        label=f"Group {grp}")
        ax.set_xlabel("Denoising fraction")
        ax.set_ylabel("Mean SAE activation")
        ax.set_title(f"Feature {fid}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Module {module_id} — top feature activation traces "
                 f"(scope={scope}, layer={layer})", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, f"module_{module_id:02d}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_module_overview(module_df: pd.DataFrame, save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(module_df)*0.5+2)))

        # Left: sizes
        ax = axes[0]
        mids   = [f"M{int(r['module_id'])}" for _, r in module_df.iterrows()]
        sizes  = module_df["n_features"].tolist()
        cs = plt.cm.tab20(np.linspace(0, 1, len(mids)))
        bars = ax.barh(mids[::-1], sizes[::-1], color=cs[::-1], height=0.7)
        ax.set_xlabel("# features in module"); ax.set_title("Module sizes")
        for b, n in zip(bars, sizes[::-1]):
            ax.text(b.get_width()+1, b.get_y()+b.get_height()/2,
                    str(n), va="center", fontsize=7)

        # Right: theme table
        ax2 = axes[1]; ax2.axis("off")
        rows = []
        for _, r in module_df.iterrows():
            kws = ", ".join(json.loads(r.get("keywords","[]"))[:4]) if r.get("keywords") else ""
            act = ", ".join(json.loads(r.get("actual_tokens","[]"))[:4])
            theme = str(r.get("theme",""))[:65]
            rows.append([f"M{int(r['module_id'])}", theme or act[:65], kws,
                         str(r.get("confidence",""))])
        if rows:
            tbl = ax2.table(cellText=rows,
                            colLabels=["Module","Theme / top tokens","Keywords","Conf"],
                            cellLoc="left", loc="center",
                            colWidths=[0.05, 0.55, 0.30, 0.10])
            tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Overview → {save_path}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Cluster annotation + activation traces")
    ap.add_argument("out_dir")
    ap.add_argument("--module-csv",        default=None)
    ap.add_argument("--vocab-labels-json", default=None)
    ap.add_argument("--records-dir",       default=None,
                    help="Phase-1 records dir for activation traces (default: out_dir)")
    ap.add_argument("--scope",             default="out_unmask")
    ap.add_argument("--trace-scope",       default="tpl_mask",
                    help="Scope for activation traces (default: tpl_mask)")
    ap.add_argument("--layer",             type=int, default=16)
    ap.add_argument("--tokenizer-path",    default=None)
    ap.add_argument("--n-features",        type=int, default=None)
    ap.add_argument("--model-name",        default=None)
    ap.add_argument("--top-per-module",    type=int, default=6)
    ap.add_argument("--topk-vocab",        type=int, default=10)
    ap.add_argument("--min-support",       type=int, default=2)
    ap.add_argument("--trace-top",         type=int, default=3,
                    help="Top-N features per module to trace (default 3)")
    ap.add_argument("--no-interp",         action="store_true")
    ap.add_argument("--no-traces",         action="store_true")
    ap.add_argument("--api-model",         default="deepseek-chat")
    ap.add_argument("--api-base",          default="https://api.deepseek.com")
    ap.add_argument("--api-key",           default=None)
    ap.add_argument("--sleep",             type=float, default=0.5)
    ap.add_argument("--save-dir",          default=None)
    args = ap.parse_args()

    base = args.save_dir or os.path.join(args.out_dir, "analysis", "annotation")
    traces_dir = os.path.join(base, "traces")
    os.makedirs(base, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    records_dir = args.records_dir or args.out_dir

    # ── 1. Module CSV ──────────────────────────────────────────────────────
    module_csv = args.module_csv
    if not module_csv:
        for candidate in [
            os.path.join(args.out_dir, "analysis", "modules", "feature_module.csv"),
            os.path.join(args.out_dir, "transcriptome", "modules", "feature_module.csv"),
            "outputs/analysis/modules/feature_module.csv",
        ]:
            if os.path.exists(candidate):
                module_csv = candidate; break
    if not module_csv or not os.path.exists(module_csv):
        print("ERROR: feature_module.csv not found. Run p5_modules.py first.")
        sys.exit(1)

    fm = pd.read_csv(module_csv)
    modules = sorted(fm["module_id"].unique())
    print(f"[p6_annotate] {len(fm)} features · {len(modules)} modules  ({module_csv!r})")

    # ── 2. Vocab labels ────────────────────────────────────────────────────
    vl_json = args.vocab_labels_json
    if not vl_json:
        for c in ["analysis_output/vocab_labels.json",
                  os.path.join(args.out_dir, "analysis_output", "vocab_labels.json")]:
            if os.path.exists(c):
                vl_json = c; break
    vocab_labels = load_cached_vocab(vl_json) if vl_json else {}
    print(f"  Vocab labels: {len(vocab_labels)} pre-computed features")

    # ── 3. Tokenizer ───────────────────────────────────────────────────────
    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path, trust_remote_code=True, local_files_only=True)
            print(f"  Tokenizer: loaded")
        except Exception as e:
            warnings.warn(f"Tokenizer failed: {e}")

    # ── 4. Actual activation data ──────────────────────────────────────────
    actual_top: dict[int, list[str]] = {}
    try:
        from cdg.analysis.io import singlecell_matrix
        Xs, obs_tok = singlecell_matrix(
            args.out_dir, scope=args.scope, layer=args.layer,
            n_features=args.n_features, model_name=args.model_name,
            tokenizer=tokenizer,
        )
        print(f"  Actual activations: {Xs.shape[0]} tokens × {Xs.shape[1]} features")
        for fid in fm["feature_id"].tolist():
            if fid >= Xs.shape[1]: continue
            col = np.asarray(Xs[:, fid].todense()).ravel()
            top_idx = np.argsort(-col)[:40]
            top_idx = top_idx[col[top_idx] > 0]
            toks = []
            for i in top_idx:
                s = str(obs_tok.iloc[int(i)].get("token_str","")).strip()
                if s and s not in _SKIP_TOKENS:
                    toks.append(s)
                if len(toks) >= 10: break
            actual_top[int(fid)] = toks
        print(f"  Actual tokens: {sum(1 for v in actual_top.values() if v)} features with data")
    except ValueError as e:
        print(f"  No singlecell data ({e})")
    except Exception as e:
        warnings.warn(f"singlecell_matrix: {e}")

    # ── 5. Per-feature table ───────────────────────────────────────────────
    feat_rows = []
    for _, row in fm.iterrows():
        fid = int(row["feature_id"]); mid = int(row["module_id"])
        vl = vocab_labels.get(fid, {})
        feat_rows.append({
            "feature_id":        fid, "module_id": mid,
            "frac_active":       row.get("frac_active",""),
            "top_tokens_logit":  json.dumps([t.get("token_str","") for t in
                                             (vl.get("top_tokens") or [])[:args.topk_vocab]],
                                             ensure_ascii=False),
            "top_tokens_actual": json.dumps(actual_top.get(fid,[])[:6], ensure_ascii=False),
        })
    pd.DataFrame(feat_rows).to_csv(os.path.join(base, "feature_labels.csv"), index=False)
    print(f"  Feature table → {os.path.join(base, 'feature_labels.csv')}")

    # ── 6. Per-module annotation + traces ──────────────────────────────────
    client = None
    if not args.no_interp:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            client = _Client(api_key, args.api_base, args.api_model)
        else:
            print("  [warn] No DEEPSEEK_API_KEY → LLM annotation skipped")

    mod_rows = []
    for mid in sorted(modules):
        mod_feats = fm[fm["module_id"]==mid].sort_values("frac_active", ascending=False)
        feat_ids  = mod_feats["feature_id"].tolist()
        top_feats = feat_ids[:args.top_per_module]

        agg = aggregate_cluster_tokens(feat_ids, vocab_labels, actual_top,
                                        topk=args.topk_vocab, min_support=args.min_support)

        feat_summaries = []
        for fid in top_feats:
            vl = vocab_labels.get(int(fid), {})
            tv = [t.get("token_str","") for t in (vl.get("top_tokens") or [])[:5]]
            at = actual_top.get(int(fid),[])[:4]
            feat_summaries.append(
                f"feat {fid}: logit=[{', '.join(repr(t) for t in tv)}]"
                + (f"  actual=[{', '.join(repr(t) for t in at)}]" if at else ""))

        act_display = [t for t, _ in agg["actual_union"][:8]]
        top_display = [t for t, _ in agg["top_union"][:6]]
        print(f"\n  Module {mid:2d} ({len(feat_ids):4d} features)  "
              f"actual_top={act_display[:5]}  logit_top={top_display[:4]}")

        theme, keywords, conf = "", [], "low"
        if client:
            r = client.call(mid, len(feat_ids), agg, feat_summaries)
            theme = r["theme"]; keywords = r["keywords"]; conf = r["confidence"]
            print(f"    → {theme[:85]!r}  [{conf}]")
            time.sleep(args.sleep)

        mod_rows.append({
            "module_id":        mid,
            "n_features":       len(feat_ids),
            "actual_tokens":    json.dumps([t for t, _ in agg["actual_union"][:15]],
                                           ensure_ascii=False),
            "top_vocab_tokens": json.dumps([t for t, _ in agg["top_union"][:10]],
                                           ensure_ascii=False),
            "theme":            theme,
            "keywords":         json.dumps(keywords, ensure_ascii=False),
            "confidence":       conf,
        })

        # Activation traces for this module
        if not args.no_traces:
            trace_feats = feat_ids[:args.trace_top]
            try:
                plot_module_traces(mid, trace_feats, records_dir,
                                   scope=args.trace_scope, layer=args.layer,
                                   save_dir=traces_dir)
            except Exception as e:
                warnings.warn(f"Trace for module {mid} failed: {e}")

    mod_df = pd.DataFrame(mod_rows)
    mod_df.to_csv(os.path.join(base, "module_labels.csv"), index=False)
    plot_module_overview(mod_df, os.path.join(base, "module_annotation.png"))
    print(f"\n  Module labels → {os.path.join(base, 'module_labels.csv')}")
    print(f"  Traces        → {traces_dir}/")
    print(f"\n[p6_annotate] Done.")


if __name__ == "__main__":
    main()
