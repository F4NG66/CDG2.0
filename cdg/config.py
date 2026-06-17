from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# topk-SAE trainer index -> target L0 (kept from original repo)
TRAINER_TO_L0 = {0: 50, 1: 80, 2: 160, 3: 320, 4: 520, 5: 820}
L0_TO_TRAINER = {v: k for k, v in TRAINER_TO_L0.items()}


@dataclass
class SAESpec:
    name: str           # local dir / logical name, e.g. "llada_mask"
    repo: str           # HF repo (for download)
    kind: str           # "mask" | "unmask"
    trainer: int = 1


# ---------------------------------------------------------------------------
# Scope = (which token region, which positions inside it, which SAE kind).
# This is the heart of the redesign: we no longer hard-code "answer region".
# A run records every scope whose region exists for a given case.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scope:
    name: str            # storage key, e.g. "tpl_mask"
    region: str          # "template" | "harm" | "output"
    pos: str             # "mask" | "unmask" | "all"  (computed per-step from x)
    sae_kind: str        # which SAEBundle.kind to encode with: "mask"|"unmask"


# Default scope set.
#   tpl_mask  -> the INJECTED BLANKS inside the prompt (PRIMARY probing/steering signal)
#   tpl_ctx   -> the framework context tokens around the blanks (needs span markers)
#   harm      -> the behavior / question text (for content-vs-injection disentangling)
#   out_*     -> the appended answer region (keeps comparability with the OLD design)
DEFAULT_SCOPES: tuple[Scope, ...] = (
    Scope("tpl_mask",   "template", "mask",   "mask"),
    Scope("tpl_ctx",    "template", "unmask", "unmask"),
    Scope("harm",       "harm",     "unmask", "unmask"),
    Scope("out_mask",   "output",   "mask",   "mask"),
    Scope("out_unmask", "output",   "unmask", "unmask"),
)


@dataclass
class RecordConfig:
    # record EARLY: the injection "decision" forms before the blanks are filled.
    record_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 1.00)
    main_fraction: float = 0.10
    entropy_top_k: int = 50
    pool: str = "mean"               # pooling over positions: "mean" | "none"
    scopes: tuple[Scope, ...] = DEFAULT_SCOPES
    decode_regions: bool = True      # also store decoded text per region/frac


@dataclass
class TemplateConfig:
    # Sentinels delimiting the framework span inside the *user content* string.
    # They are removed before the model sees the prompt; used only to locate
    # the template token span. If absent, we fall back to "all input masks".
    open_tag: str = "<<TPL>>"
    close_tag: str = "<</TPL>>"
    # Marker the schema uses for N blanks; expanded to mask_token * N at build time.
    mask_marker_re: str = r"<mask:(\d+)>"
    behavior_placeholder: str = "{behavior}"


@dataclass
class DecodeConfig:
    gen_length: int = 128
    steps: int = 64
    block_length: Optional[int] = None
    temperature: float = 0.0
    cfg_scale: float = 0.0
    # NEW: if True, denoise EVERY mask in the sequence (injected-input + appended-
    # output), not just the appended block. Required for template-injection.
    fill_all_masks: bool = True


@dataclass
class BackendConfig:
    name: str
    model_id: str
    saes: list[SAESpec]
    record_layers: tuple[int, ...]
    remask: str = "low_confidence"
    mask_id: Optional[int] = None
    layer_to_block_offset: int = 0
    blocks_attr: Optional[str] = None
    mask_token: str = "<|mdm_mask|>"     # text form of the mask, for prompt building
    record: RecordConfig = field(default_factory=RecordConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)


# ---------------------------------------------------------------------------
# Presets.  The *_attack variants align decoding knobs with the adversarial
# reference harness (steps=128, gen_length=128, single block, temperature=0.2,
# mask_id=126336) so our numbers are comparable to the attacker's.
# ---------------------------------------------------------------------------
def llada_config() -> BackendConfig:
    return BackendConfig(
        name="llada",
        model_id="GSAI-ML/LLaDA-8B-Instruct",
        saes=[
            SAESpec("llada_mask",   "AwesomeInterpretability/llada-mask-topk-sae",
                    "mask", 1),
            SAESpec("llada_unmask", "AwesomeInterpretability/llada-unmask-topk-sae",
                    "unmask", 1),
        ],
        record_layers=(11, 16, 26),
        remask="low_confidence",
        mask_id=126336,
        layer_to_block_offset=0,
        mask_token="<|mdm_mask|>",
    )


def llada_attack_config() -> BackendConfig:
    """LLaDA with decoding aligned to the attack reference harness."""
    c = llada_config()
    c.name = "llada"
    c.decode = DecodeConfig(gen_length=128, steps=128, block_length=128,
                            temperature=0.2, fill_all_masks=True)
    return c


def dream_config() -> BackendConfig:
    return BackendConfig(
        name="dream",
        model_id="Dream-org/Dream-v0-Instruct-7B",
        saes=[
            SAESpec("dream_mask",   "AwesomeInterpretability/dlm-mask-topk-sae",
                    "mask", 1),
            SAESpec("dream_unmask", "AwesomeInterpretability/dlm-unmask-topk-sae",
                    "unmask", 1),
        ],
        record_layers=(5, 14, 23),
        remask="entropy",
        mask_id=None,
        layer_to_block_offset=0,
        mask_token="<|mask|>",
    )


def dream_attack_config() -> BackendConfig:
    c = dream_config()
    c.decode = DecodeConfig(gen_length=128, steps=128, block_length=128,
                            temperature=0.2, fill_all_masks=True)
    return c


PRESETS = {
    "llada": llada_config,
    "llada_attack": llada_attack_config,
    "dream": dream_config,
    "dream_attack": dream_attack_config,
}


def get_backend_config(name: str) -> BackendConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown backend '{name}', choose from: {list(PRESETS)}")
    return PRESETS[name]()
