"""M2 confirmatory DIJA construction - FROZEN deterministic seed policy (v2).

GENERALIZED_SAFETY_V2_M2_DIJA_SEED_POLICY_V2. Supersedes v1 (preserved unmodified as
provenance in generalized_safety_v2_m2_dija_construction_protocol_v1). v1's formal
non-confirmatory smoke FAILED determinism (GPU kernel non-determinism, not seeding).
v2 = v1 seed identity rules UNCHANGED + frozen deterministic runtime controls.
This is a pre-confirmatory reproducibility/runtime amendment - not outcome-driven, not
an attack-strength or sampling-distribution change; no confirmatory DIJA output existed.

Freezes WHICH stochastic draw is used per Qwen generation call. It does NOT
change the sampling distribution: Qwen2.5-7B-Instruct keeps do_sample=True,
temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05 from its
generation_config.json, exactly as in M1.

Frozen rules
------------
namespace      : GENERALIZED_SAFETY_V2_M2_CONFIRMATORY_DIJA_CONSTRUCTION_V1
call id format : "M2DIJA|<source_id>|<branch>|cand<NN>|call<NN>"
                 source_id  - frozen source id (FSBV1 fresh_group_id for confirmatory)
                 branch     - HARMFUL_B (harmful source -> B) | BENIGN_C (benign source -> C)
                 cand<NN>   - refinement candidate index, 2-digit zero-padded, 1-based
                 call<NN>   - stochastic call ordinal within the candidate, 2-digit, 1-based
                 M1 performs exactly ONE stochastic generation per source per branch and
                 has no candidates, so confirmatory calls are always cand01 / call01.
                 Components must match ^[A-Za-z0-9_]+$ (no separator can leak in).
seed material  : namespace + "::" + call_id
byte encoding  : UTF-8
hash           : SHA-256
integer rule   : int.from_bytes(digest[0:4], byteorder="big", signed=False)
range rule     : no modulo; natural range [0, 2**32 - 1]; the same integer seeds
                 torch (CPU + every CUDA device), Python `random` and NumPy
                 (NumPy requires < 2**32, which this range satisfies).
application    : transformers 4.38.2 GenerationMixin.generate() has no `generator`
                 argument, so the per-call torch.Generator path is unavailable. The
                 fallback is used: apply_seed(seed) is called immediately before each
                 serial Refiner.qwen_generate() call (tokenization inside it consumes
                 no RNG, so this is equivalent to seeding immediately before
                 model.generate()). Calls are strictly serial, batch size 1.
deterministic runtime (v2 addition, frozen):
                 env  CUBLAS_WORKSPACE_CONFIG=:4096:8   (set before process start)
                 torch.use_deterministic_algorithms(True)
                 torch.backends.cuda.matmul.allow_tf32 = False
                 torch.backends.cudnn.allow_tf32 = False
                 torch.backends.cudnn.benchmark = False
                 configure_deterministic_runtime() must run before the model is loaded.
"""
import hashlib
import random
import re

NAMESPACE = "GENERALIZED_SAFETY_V2_M2_CONFIRMATORY_DIJA_CONSTRUCTION_V1"
CALL_ID_PREFIX = "M2DIJA"
SEPARATOR = "::"
ENCODING = "utf-8"
BRANCHES = ("HARMFUL_B", "BENIGN_C")
_COMPONENT = re.compile(r"^[A-Za-z0-9_]+$")


def make_call_id(source_id: str, branch: str, candidate: int = 1, call_ordinal: int = 1) -> str:
    if not _COMPONENT.match(source_id):
        raise ValueError(f"illegal source_id component: {source_id!r}")
    if branch not in BRANCHES:
        raise ValueError(f"illegal branch: {branch!r}")
    for name, v in (("candidate", candidate), ("call_ordinal", call_ordinal)):
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 99:
            raise ValueError(f"illegal {name}: {v!r}")
    return f"{CALL_ID_PREFIX}|{source_id}|{branch}|cand{candidate:02d}|call{call_ordinal:02d}"


def seed_material(call_id: str, namespace: str = NAMESPACE) -> bytes:
    return (namespace + SEPARATOR + call_id).encode(ENCODING)


def derive_seed(call_id: str, namespace: str = NAMESPACE) -> int:
    digest = hashlib.sha256(seed_material(call_id, namespace)).digest()
    return int.from_bytes(digest[0:4], byteorder="big", signed=False)


def apply_seed(seed: int) -> None:
    """Seed every RNG that can participate in the generation path."""
    import numpy as np
    import torch

    if not 0 <= seed < 2**32:
        raise ValueError(f"seed out of frozen range: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DETERMINISTIC_RUNTIME = {
    "env.CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "torch.use_deterministic_algorithms": True,
    "torch.backends.cuda.matmul.allow_tf32": False,
    "torch.backends.cudnn.allow_tf32": False,
    "torch.backends.cudnn.benchmark": False,
}


def configure_deterministic_runtime() -> dict:
    """Apply and verify the frozen deterministic runtime. Call before loading the model."""
    import os
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' in the launch environment")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return read_deterministic_runtime()


def read_deterministic_runtime() -> dict:
    import os
    import torch

    return {
        "env.CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch.use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch.backends.cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch.backends.cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch.backends.cudnn.benchmark": torch.backends.cudnn.benchmark,
    }


def policy_descriptor() -> dict:
    return {
        "namespace": NAMESPACE,
        "call_id_format": "M2DIJA|<source_id>|<branch>|cand<NN>|call<NN>",
        "call_id_component_regex": _COMPONENT.pattern,
        "branches": list(BRANCHES),
        "seed_material": "namespace + '::' + call_id",
        "byte_encoding": "UTF-8",
        "hash_algorithm": "SHA-256",
        "integer_extraction": "int.from_bytes(digest[0:4], 'big', signed=False)",
        "range_rule": "no modulo; [0, 2**32-1]",
        "rngs_seeded": ["python.random", "numpy.random", "torch (CPU + all CUDA via manual_seed and cuda.manual_seed_all)"],
        "application": "global RNG seeding immediately before each serial Refiner.qwen_generate() call (per-call torch.Generator unsupported by transformers 4.38.2 generate())",
        "confirmatory_calls_per_group": 2,
        "confirmatory_candidate_and_ordinal": "cand01 / call01 only (M1 has one stochastic call per source per branch, no candidates, no rerolls)",
        "policy_version": "GENERALIZED_SAFETY_V2_M2_DIJA_SEED_POLICY_V2",
        "seed_identity_rules_changed_from_v1": False,
        "deterministic_runtime": DETERMINISTIC_RUNTIME,
        "deterministic_runtime_applied": "configure_deterministic_runtime() before model load; env var set in sbatch before process start",
    }
