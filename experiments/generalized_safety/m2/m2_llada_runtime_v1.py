"""M2 confirmatory LLaDA runtime (shared by smoke and confirmatory executors).

- Frozen bindings: every code/data artifact is SHA-verified before use.
- Betas: read ONLY through the frozen key-mapping artifact from the frozen M1 / M2 beta files.
- Generation paths:
    BASELINE, M1_ADD_*  -> historical m1.generate_one, unmodified (SHA-bound runner)
    M2_DMNP_*           -> generate_one_m2dmnp: frozen patched copy of generate_one, exec'd in the
                           historical runner's own module namespace (same globals/constants/helpers)
- Hooks: M1 SafetyLayerLocationHookBank and frozen M2NormPreservingHookBank run AS-IS.
- Instrumentation: read-only L16Observer (returns None from every hook, never alters activations).
"""
import hashlib, importlib.util, inspect, json, math
from pathlib import Path

import torch

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"
HERE = DEV / "generalized_safety_v2_m2_llada_executor_v1"
MODEL = Path(__import__("os").environ["CDG_LLADA_MODEL"])
MASK_ID = 126336
LAYER = 16

P = {
    "runner": ROOT / "scripts/rrae_development/run_generalized_safety_m1_eval_v1.py",
    "compat_helper": DEV / "generalized_safety_v2_llada_baseline_dija_localization_compat_v1/frozen_generation_helpers_compat_v1.py",
    "directions": DEV / "generalized_safety_v2_directions_v1/GENERALIZED_SAFETY_V2_DIRECTIONS.pt",
    "m1_betas": DEV / "generalized_safety_v2_beta_calibration_v1/BETAS.json",
    "m2_betas": DEV / "generalized_safety_v2_m2_beta_freeze_v1/M2_BETAS.json",
    "m2_beta_freeze": DEV / "generalized_safety_v2_m2_beta_freeze_v1/M2_BETA_FREEZE.json",
    "beta_mapping": DEV / "generalized_safety_v2_m2_beta_key_mapping_v1/M2_BETA_KEY_MAPPING.json",
    "m2_hook": DEV / "generalized_safety_v2_m2_contract_v1/m2_norm_preserving_hook_v1.py",
    "patched_generate": HERE / "generate_one_m2dmnp_source.py",
    "hard_failure_policy": DEV / "generalized_safety_v2_m2_llada_hard_failure_policy_v1/M2_LLADA_HARD_FAILURE_POLICY.json",
    "norm_clarification": DEV / "generalized_safety_v2_m2_llada_hard_failure_policy_clarification_v1/M2_NORM_THRESHOLD_CLARIFICATION.json",
}
EXPECT = {
    "runner": "3ff21e59c7d045c42187d9880410280e92d2efcd6a585ecad1db984b63da618f",
    "hist": "cca5bb26111932c78815ef5a75bdc4898ce9a1a023550c1b9a06dbcbacdae224",
    "baseline_helper": "fa5ad065c0c1e8b69228bc12356e8ee1cdb10c38ea864d2e21f3d8bdabbfae62",
    "canonical_helper": "67832c5cb193699b2fe83b545b6013b290dc8cddbd8d35504c983bda1cb92680",
    "compat_helper": "98189fd24be8789891507c19ab1d9f630b5ec5dd5c85e74b8b76b97ccdae577a",
    "directions": "e43916ff3aaab143201daeeae78c66fa50830b0b4a7ff4ff6447fb5b3c05d487",
    "m1_betas": "158efb559f2fdb40b5c581a5d1322fe925704bd6b83e6a73041addd425002492",
    "beta_mapping": "fcbb5e2e3294b9f9166ab4f52f039356dd23fab464dd05245627e373ef16704c",
    "m2_hook": "20370c45607b8acea040bc1aa5a1b3355d301a7d0eeadfdadcadfce0c48e49d1",
    "patched_generate": "45693f610249d02a8336957f8088d2f83b91e796e1266c47accb740702a3bad2",
    "hard_failure_policy": "efe2e4a8d82052c9ac8dca8fadb746dbce3c0c7f7ea2980db04bd1e6442493fe",
    "norm_clarification": "a6ee0d64a76fe4e0a04f6ae1b0eb636199efce6a4203526b409168711709a98d",
    "modeling_llada": "98bac7e53fef0bb7ca01e3716c11a7f710d183e10dbb9783b88db9dbba2e3766",
}
DIRECTION_VECTOR_SHA256 = {  # sha256 of float32 contiguous bytes, frozen in M2_CONTRACT_V1
    "V_ALL": "937a78750761e63ebc1362bd43e7b3adfa20790b6cec9ae15df01a7befe177e7",
    "V_DIJA": "2a52b08871548bdcb951ea5a32595fc30d7fbc4d057e51a0f3f9c39d8140c6b5",
    "V_RENELLM": "27cd930d1633cd84d1fb8c33b98f16a6dd86220cadf4d5fcd90c404cb9f7be33",
}
# human-readable cross-check ONLY (never used as a source of truth; divergence => preflight failure)
M1_BETA_REFERENCE = {"V_DIJA": 74.77510070800781, "V_RENELLM": 81.59530639648438, "V_ALL": 73.11751556396484}
DIRECTIONS = ("V_ALL", "V_DIJA", "V_RENELLM")
ARMS = ("BASELINE", "M1_ADD_V_ALL", "M1_ADD_V_DIJA", "M1_ADD_V_RENELLM",
        "M2_DMNP_V_ALL", "M2_DMNP_V_DIJA", "M2_DMNP_V_RENELLM")
M2_DENOMINATOR_INVALID = 1e-8   # ||h_tilde|| <= this -> category C (M2_NUMERICAL_DENOMINATOR_INVALID)
M2_NORM_ANOMALY_SENTINEL = 1.0  # ||h_tilde|| <  this -> category C (M2_NORM_ANOMALY_SENTINEL)
M2_FP32_REL_NORM_DRIFT_MAX = 1e-5  # hook's own fp32 audit (pre-bf16-cast) invariant


def arm_parts(arm):
    if arm == "BASELINE":
        return "BASELINE", None
    method, _, d = arm.partition("_V_")
    return {"M1_ADD": "M1_ADD", "M2_DMNP": "M2_DMNP"}[method], "V_" + d


class CategoryCFailure(RuntimeError):
    def __init__(self, label, detail):
        super().__init__(f"CATEGORY_C_FAIL_CLOSED {label}: {detail}")
        self.label, self.detail = label, detail


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vector_sha(t):
    return hashlib.sha256(t.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- bindings
def verify_file_bindings():
    for k in ("runner", "compat_helper", "directions", "m1_betas", "beta_mapping", "m2_hook",
              "patched_generate", "hard_failure_policy", "norm_clarification"):
        got = sha_file(P[k])
        if got != EXPECT[k]:
            raise RuntimeError(f"BINDING_SHA_MISMATCH {k}: {got}")
    mp = json.loads(P["beta_mapping"].read_text())
    if mp["m2_beta_file_sha256"] != sha_file(P["m2_betas"]) or mp["m2_beta_freeze_sha256"] != sha_file(P["m2_beta_freeze"]):
        raise RuntimeError("BINDING_SHA_MISMATCH m2 beta files vs mapping")
    if mp["m1_beta_file_sha256"] != sha_file(P["m1_betas"]):
        raise RuntimeError("BINDING_SHA_MISMATCH m1 beta file vs mapping")
    return mp


def load_betas():
    """Returns {'M1_ADD': {d: beta}, 'M2_DMNP': {d: beta}} read through the frozen mapping."""
    mp = verify_file_bindings()
    m1f, m2f = json.loads(P["m1_betas"].read_text()), json.loads(P["m2_betas"].read_text())
    out = {"M1_ADD": {}, "M2_DMNP": {}}
    for d in DIRECTIONS:
        out["M1_ADD"][d] = float(m1f[mp["mapping"][d]["m1_key"]])
        out["M2_DMNP"][d] = float(m2f[mp["mapping"][d]["m2_key"]])
    for d in DIRECTIONS:
        if out["M1_ADD"][d] != M1_BETA_REFERENCE[d]:
            raise RuntimeError(f"M1 beta for {d} diverges from reference cross-check: {out['M1_ADD'][d]}")
        for m in out:
            if not math.isfinite(out[m][d]) or out[m][d] <= 0:
                raise RuntimeError(f"invalid beta {m} {d}")
    return out, mp


def load_historical():
    """Historical runner + its frozen hook/helper modules, wired exactly as the M1 held-out executor did."""
    verify_file_bindings()
    m1 = load_module("historical_m1", P["runner"])
    for attr, key in (("HIST_PATH", "hist"), ("BASELINE_PATH", "baseline_helper"), ("HELPER_PATH", "canonical_helper")):
        if sha_file(getattr(m1, attr)) != EXPECT[key]:
            raise RuntimeError(f"BINDING_SHA_MISMATCH {key}")
    hist = load_module("historical_hook", Path(m1.HIST_PATH))
    m1.hist = hist
    baseline_helper = load_module("historical_baseline_helper", Path(m1.BASELINE_PATH))
    helper = load_module("historical_dija_helper", Path(m1.HELPER_PATH))
    compat_helper = load_module("frozen_dija_compat_helper", P["compat_helper"])
    if hasattr(m1, "sha_text"):
        helper.sha256_text = m1.sha_text
        compat_helper.sha256_text = m1.sha_text
    # frozen M2 generate function, compiled in the historical runner's namespace
    src = P["patched_generate"].read_text()
    exec(compile(src, str(P["patched_generate"]), "exec"), m1.__dict__)
    m2hook = load_module("m2_norm_preserving_hook_v1", P["m2_hook"])
    return {"m1": m1, "hist": hist, "baseline": baseline_helper, "helper": helper,
            "compat_helper": compat_helper, "m2hook": m2hook}


def load_directions():
    d = torch.load(P["directions"], map_location="cpu", weights_only=True)
    assert d["schema"] == "GENERALIZED_SAFETY_V2_DIRECTIONS_V1"
    assert int(d["layer_zero_based"]) == LAYER and d["orientation"] == "safe_minus_harmful"
    mp = json.loads(P["beta_mapping"].read_text())["mapping"]
    vecs = {}
    for k in DIRECTIONS:
        v = d[mp[k]["direction_tensor_key"]].detach().float().view(-1)
        assert v.numel() == 4096 and torch.isfinite(v).all()
        assert math.isclose(float(v.norm()), 1.0, abs_tol=1e-5), k
        if vector_sha(v) != DIRECTION_VECTOR_SHA256[k]:
            raise RuntimeError(f"DIRECTION_SHA_MISMATCH {k}")
        vecs[k] = v
    return vecs


# ---------------------------------------------------------------- step-audit policies
def M1_STEP_AUDIT_POLICY(m1):
    """Verbatim re-statement of the M1 audit block removed from the patched copy (loop-equivalence test only)."""
    require = m1.require

    def policy(*, item, step_audits, trajectory_steps):
        beta = float(item["beta"])
        tolerance = max(0.25, 0.02 * abs(beta))

        for audit in step_audits:
            require(
                abs(float(audit["target_mean_delta_l2"]) - beta)
                <= tolerance,
                "M1 delta magnitude audit failed",
            )

            require(
                float(audit["target_min_cosine_to_v_safety"]) >= 0.995,
                "M1 direction cosine audit failed",
            )

            require(
                float(audit["non_target_max_abs_delta"]) == 0.0,
                "Non-target hidden state changed",
            )
    return policy


def M2_STEP_AUDIT_POLICY(*, item, step_audits, trajectory_steps):
    """M2 invariants on the frozen hook's own per-step audit records. Any violation -> category C."""
    if len(step_audits) != trajectory_steps:
        raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", "M2 audit count != trajectory steps")
    for i, a in enumerate(step_audits, 1):
        if int(a["step"]) != i or int(a["layer_zero_based"]) != LAYER or a["scope"] != "current_mask":
            raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", f"M2 step/layer/scope mismatch at {i}")
        if float(a["eps"]) != 1e-8:
            raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", "M2 eps drift")
        if float(a["non_target_max_abs_delta"]) != 0.0:
            raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", f"M2 non-target changed at step {i}")
        vals = [float(a[k]) for k in a if isinstance(a[k], (int, float)) and not isinstance(a[k], bool)]
        if not all(math.isfinite(v) for v in vals):
            raise CategoryCFailure("NONFINITE_VALUE", f"M2 hook audit non-finite at step {i}")
        if float(a["target_max_rel_norm_drift"]) > M2_FP32_REL_NORM_DRIFT_MAX:
            raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", f"M2 fp32 norm drift {a['target_max_rel_norm_drift']} at step {i}")


# ---------------------------------------------------------------- read-only observer
class L16Observer:
    """Read-only instrumentation. Registered as:
         model forward-pre-hook          -> step counter + independent current_mask (input_ids == MASK_ID)
         block[16] forward hook (FIRST)  -> snapshot of the un-steered block output h
         block[16] forward hook (LAST)   -> the output after every steering bank (h_new)
       Every hook returns None, so activations are never altered."""

    def __init__(self, model, block, output_to_hidden):
        self._to_hidden = output_to_hidden
        self.enabled = True
        self.capture_steps = set()
        self.row = None
        self._h_model = model.register_forward_pre_hook(self._model_pre)
        self._h_pre = block.register_forward_hook(self._block_first)
        self._h_post = None

    def attach_post(self, block):  # call AFTER all steering banks are constructed
        self._h_post = block.register_forward_hook(self._block_last)

    def close(self):
        for h in (self._h_model, self._h_pre, self._h_post):
            if h is not None:
                h.remove()

    # ---- row lifecycle
    def begin_row(self, *, method, direction_key, vector, beta, bank):
        if method != "BASELINE":
            if not torch.isfinite(vector).all():
                raise CategoryCFailure("NONFINITE_VALUE", "direction non-finite")
            if not math.isfinite(beta):
                raise CategoryCFailure("NONFINITE_VALUE", "beta non-finite")
        self.row = {"method": method, "direction_key": direction_key, "beta": beta, "bank": bank,
                    "v": None if vector is None else vector.to("cuda", torch.float32).view(1, -1)}
        self.step = 0
        self.mask = None
        self.pre = None
        self.captures = []
        self.per_step = []
        self.agg = {"hooked_steps": 0, "affected_token_count": 0, "target_positions_unchanged": 0,
                    "min_pre_norm": math.inf, "min_raw_additive_norm": math.inf,
                    "min_post_norm": math.inf if method == "M2_DMNP" else None,
                    "max_rel_norm_preservation_error": 0.0 if method == "M2_DMNP" else None,
                    "nan_count": 0, "inf_count": 0,
                    "anomaly_sentinel_count": 0 if method == "M2_DMNP" else None,
                    "denominator_invalid_count": 0 if method == "M2_DMNP" else None,
                    "hook_invariant_failure_count": 0}

    def end_row(self, *, trajectory_steps):
        a = dict(self.agg)
        if self.enabled:
            if self.step != trajectory_steps or a["hooked_steps"] != trajectory_steps:
                self._fail("HOOK_INVARIANT_VIOLATION", f"forward/hook count {self.step}/{a['hooked_steps']} != {trajectory_steps}")
            if a["affected_token_count"] == 0 and self.row["method"] != "BASELINE":
                self._fail("HOOK_INVARIANT_VIOLATION", "no affected tokens")
        for k in ("min_pre_norm", "min_raw_additive_norm", "min_post_norm"):
            if a[k] == math.inf:
                a[k] = None
        a["per_step_min_raw_additive_norm"] = [s["min_raw_additive_norm"] for s in self.per_step]
        a["observer_enabled"] = self.enabled
        return a

    def _fail(self, label, detail):
        self.agg["hook_invariant_failure_count"] += label == "HOOK_INVARIANT_VIOLATION"
        raise CategoryCFailure(label, detail)

    # ---- hooks (all return None)
    def _model_pre(self, _module, args):
        if not self.enabled or self.row is None:
            return None
        self.step += 1
        self.mask = (args[0] == MASK_ID).detach().clone()
        return None

    def _block_first(self, _module, _inputs, output):
        if not self.enabled or self.row is None:
            return None
        self.pre = self._to_hidden(output).detach().clone()
        return None

    def _block_last(self, _module, _inputs, output):
        if not self.enabled or self.row is None:
            return None
        with torch.no_grad():
            self._observe(self._to_hidden(output).detach())
        return None

    def _observe(self, post):
        r, A = self.row, self.agg
        pre = self.pre
        A["hooked_steps"] += 1
        nan = int(torch.isnan(pre).sum() + torch.isnan(post).sum())
        inf = int(torch.isinf(pre).sum() + torch.isinf(post).sum())
        A["nan_count"] += nan
        A["inf_count"] += inf
        if nan or inf:
            raise CategoryCFailure("NONFINITE_VALUE", f"step {self.step}: nan={nan} inf={inf}")
        target = self.mask.to(post.device)
        if list(target.shape) != list(post.shape[:2]):
            self._fail("HOOK_INVARIANT_VIOLATION", "mask/hidden shape mismatch")
        changed = (post.float() - pre.float()).abs().amax(-1) > 0
        if r["method"] == "BASELINE":
            if bool(changed.any()):
                self._fail("HOOK_INVARIANT_VIOLATION", f"baseline activation changed at step {self.step}")
            self.per_step.append({"step": self.step, "min_raw_additive_norm": None})
            return
        bank = r["bank"]
        if bank.global_step != self.step:
            self._fail("HOOK_INVARIANT_VIOLATION", f"bank step {bank.global_step} != observer step {self.step}")
        if not torch.equal(bank.current_mask.to(target.device, torch.bool), target):
            self._fail("HOOK_INVARIANT_VIOLATION", "bank current_mask != input_ids == MASK_ID")
        if int(target.sum()) == 0:
            self._fail("HOOK_INVARIANT_VIOLATION", "empty target")
        if bool((changed & ~target).any()):
            self._fail("HOOK_INVARIANT_VIOLATION", f"non-target position changed at step {self.step}")
        h = pre.float()[target]
        hn = torch.linalg.vector_norm(h, dim=-1)
        raw = h + r["beta"] * r["v"]
        rn = torch.linalg.vector_norm(raw, dim=-1)
        if not (torch.isfinite(hn).all() and torch.isfinite(rn).all()):
            raise CategoryCFailure("NONFINITE_VALUE", f"norm non-finite at step {self.step}")
        n_t = int(target.sum())
        A["affected_token_count"] += n_t
        A["target_positions_unchanged"] += int((~changed & target).sum())
        A["min_pre_norm"] = min(A["min_pre_norm"], float(hn.min()))
        A["min_raw_additive_norm"] = min(A["min_raw_additive_norm"], float(rn.min()))
        rec = {"step": self.step, "target_count": n_t, "min_pre_norm": float(hn.min()),
               "min_raw_additive_norm": float(rn.min())}
        if r["method"] == "M2_DMNP":
            pn = torch.linalg.vector_norm(post.float()[target], dim=-1)
            if not torch.isfinite(pn).all():
                raise CategoryCFailure("NONFINITE_VALUE", f"post norm non-finite at step {self.step}")
            rel = float(((pn - hn).abs() / hn).max())
            A["min_post_norm"] = min(A["min_post_norm"], float(pn.min()))
            A["max_rel_norm_preservation_error"] = max(A["max_rel_norm_preservation_error"], rel)
            rec.update({"min_post_norm": float(pn.min()), "max_rel_norm_preservation_error": rel})
            if float(rn.min()) <= M2_DENOMINATOR_INVALID:
                A["denominator_invalid_count"] += 1
                self.per_step.append(rec)
                raise CategoryCFailure("M2_NUMERICAL_DENOMINATOR_INVALID", f"step {self.step}: min ||h_tilde||={float(rn.min())}")
            if float(rn.min()) < M2_NORM_ANOMALY_SENTINEL:
                A["anomaly_sentinel_count"] += 1
                self.per_step.append(rec)
                raise CategoryCFailure("M2_NORM_ANOMALY_SENTINEL", f"step {self.step}: min ||h_tilde||={float(rn.min())}")
        self.per_step.append(rec)
        if self.step in self.capture_steps:  # smoke only: raw tensors for independent hand-computation
            self.captures.append({"step": self.step, "pre_target_f32": h.cpu(), "post_target_f32": post.float()[target].cpu(),
                                  "record": rec})


# ---------------------------------------------------------------- model/bank setup
def build_runtime():
    H = load_historical()
    betas, mapping = load_betas()
    vecs = load_directions()
    from transformers import AutoModel, AutoTokenizer
    assert MODEL.is_dir() and torch.cuda.is_available()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    assert H["helper"].tokenize_without_specials(tok, "<|mdm_mask|>") == [MASK_ID]
    model = AutoModel.from_pretrained(str(MODEL), trust_remote_code=True, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, local_files_only=True).eval().to(device)
    assert model.dtype == torch.bfloat16
    if sha_file(Path(inspect.getfile(type(model)))) != EXPECT["modeling_llada"]:
        raise RuntimeError("BINDING_SHA_MISMATCH modeling_llada")
    hist = H["hist"]
    blocks, blocks_attr, _ = hist.resolve_transformer_blocks(model)
    assert len(blocks) == 32
    obs = L16Observer(model, blocks[LAYER], hist.output_to_hidden)          # FIRST on block 16
    m1_banks = {k: hist.SafetyLayerLocationHookBank(blocks, v, (LAYER,)) for k, v in vecs.items()}
    m2_banks = {k: H["m2hook"].M2NormPreservingHookBank(blocks, v, (LAYER,), hist.output_to_hidden,
                                                         hist.output_with_hidden) for k, v in vecs.items()}
    obs.attach_post(blocks[LAYER])                                          # LAST on block 16
    return {"H": H, "betas": betas, "mapping": mapping, "vecs": vecs, "tok": tok, "model": model,
            "device": device, "obs": obs, "m1_banks": m1_banks, "m2_banks": m2_banks,
            "blocks_attr": blocks_attr}


def reset_all_banks(rt):
    for b in list(rt["m1_banks"].values()) + list(rt["m2_banks"].values()):
        b.reset_run(layer=None, beta=0.0, active_steps=[])


def run_item(rt, item, *, variant="WRAPPED"):
    """variant: WRAPPED (confirmatory path) | UNWRAPPED (observer disabled; historical-executor-equivalent)
               | COPY_LOOP_M1 (patched copy + M1 banks + verbatim M1 audit; loop-equivalence test only)."""
    H, obs = rt["H"], rt["obs"]
    m1 = H["m1"]
    method, d = arm_parts(item["arm"])
    helper = H["compat_helper"] if item["input_helper_route"] == "DIJA_COMPAT" else H["helper"]
    if item["input_helper_route"] not in ("DIJA_COMPAT", "DIJA_ORIGINAL", "RENELLM_STANDARD"):
        raise RuntimeError(item["input_helper_route"])
    reset_all_banks(rt)
    if method == "M2_DMNP" or variant == "COPY_LOOP_M1":
        banks = rt["m2_banks"] if method == "M2_DMNP" else rt["m1_banks"]
    else:
        banks = rt["m1_banks"]
    bank = None if d is None else banks[d]
    obs.enabled = variant != "UNWRAPPED"
    obs.begin_row(method=method, direction_key=d, vector=None if d is None else rt["vecs"][d],
                  beta=float(item["beta"]), bank=bank)
    try:
        if method == "M2_DMNP":
            res = m1.generate_one_m2dmnp(item=item, model=rt["model"], tokenizer=rt["tok"], helper=helper,
                                         baseline=H["baseline"], banks=banks, device=rt["device"],
                                         step_audit_policy=M2_STEP_AUDIT_POLICY)
        elif variant == "COPY_LOOP_M1":
            res = m1.generate_one_m2dmnp(item=item, model=rt["model"], tokenizer=rt["tok"], helper=helper,
                                         baseline=H["baseline"], banks=banks, device=rt["device"],
                                         step_audit_policy=M1_STEP_AUDIT_POLICY(m1))
        else:
            res = m1.generate_one(item=item, model=rt["model"], tokenizer=rt["tok"], helper=helper,
                                  baseline=H["baseline"], banks=banks, device=rt["device"])
        inst = obs.end_row(trajectory_steps=int(res["trajectory_steps"]))
    finally:
        obs.row = None
        obs.enabled = True
    if res["initial_ids_sha256"] != item["expected_initial_ids_sha256"]:
        raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", "initial ids drift vs CPU manifest")
    if int(res["trajectory_steps"]) != int(item["expected_trajectory_steps"]):
        raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", "trajectory steps drift vs CPU manifest")
    if int(res["generation_seed"]) != int(item["generation_seed"]):
        raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", "seed drift")
    if method == "BASELINE":
        if res["vector_key"] is not None or int(res["application_count"]) != 0 or res["step_audits"] != []:
            raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", "baseline applied steering")
    elif int(res["application_count"]) != int(res["trajectory_steps"]):
        raise CategoryCFailure("HOOK_INVARIANT_VIOLATION", "persistent application mismatch")
    res.update({"arm": item["arm"], "method": method, "direction": d, "variant": variant,
                "source_case_id": item["source_case_id"], "v2_condition": item["v2_condition"],
                "historical_condition": item["condition"], "group_key": item["group_key"],
                "group_ordinal": item["group_ordinal"], "split": item["split"],
                "beta_source": item["beta_source"], "instrumentation": inst})
    if method == "M2_DMNP":
        res.update({"m2_eps": 1e-8, "m2_equation": "h_new = ||h||_2 * h_tilde / (||h_tilde||_2 + eps), h_tilde = h + beta*v"})
    return res
