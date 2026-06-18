"""cdg.analysis – analysis functions + transcriptomics-style SAE analysis.

Existing facade for probe/interpret/steering (backward-compatible).
New io layer: pseudobulk_matrix, singlecell_matrix (from cdg.analysis.io).
"""
from ..probe import (
    load_records,
    top_separating_features,
    top_diff_features,
    diff_feature_matrix,
    feature_atlas,
    probe_sweep,
    save_analysis,
)
from ..interpret import (
    feature_vocab_labels,
    feature_atlas_data,
    coactivation_matrix,
    mechanism_summary,
    print_feature_report,
    save_atlas,
)
from ..steering import (
    build_steering_vectors,
    build_did_vectors,
    SteeringVectors,
)


def delta_vector(out_dir, model_name, *, scope="tpl_mask", space="hidden",
                 frac=0.10, pos_groups=("B",), neg_groups=("C",), **kw):
    """B-C mean-difference steering vector (alias for build_steering_vectors)."""
    return build_steering_vectors(out_dir, model_name, scope=scope, space=space,
                                  frac=frac, pos_groups=pos_groups,
                                  neg_groups=neg_groups, **kw)


def steering_defense(out_dir, model_name, *, scope="tpl_mask", space="hidden",
                     frac=0.10, **kw):
    """Convenience wrapper: build direction vector + return as runner dict."""
    sv = build_steering_vectors(out_dir, model_name, scope=scope, space=space,
                                frac=frac, **kw)
    return sv.as_runner_dict(normalize=True)


from .io import (
    load_manifest,
    load_record,
    pseudobulk_matrix,
    singlecell_matrix,
)

__all__ = [
    "load_records",
    "top_separating_features", "top_diff_features", "diff_feature_matrix",
    "feature_atlas", "probe_sweep", "save_analysis",
    "feature_vocab_labels", "feature_atlas_data", "coactivation_matrix",
    "mechanism_summary", "print_feature_report", "save_atlas",
    "build_steering_vectors", "build_did_vectors", "SteeringVectors",
    "delta_vector", "steering_defense",
    # transcriptomics io
    "load_manifest", "load_record", "pseudobulk_matrix", "singlecell_matrix",
]
