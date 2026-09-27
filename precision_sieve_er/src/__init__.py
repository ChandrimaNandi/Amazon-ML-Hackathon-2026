"""
Precision-Sieve Entity Resolution Package.
Pillars:
1. Zero-Tolerance Deterministic Hard Veto Gates
2. Asymmetric 4x False-Positive Loss GBDT Ensemble
3. High-Precision Feature Engineering
4. Multi-Model Tabular Rank Averaging
5. Ambiguity Rejection Engine & Margin Gating
"""

from precision_sieve_er.src.normalization import (
    clean_entity_field,
    normalize_text,
    transliterate_to_latin,
    create_normalized_dataframe
)

from precision_sieve_er.src.veto_gates import (
    extract_numbers,
    extract_pin_codes,
    extract_street_numbers,
    extract_us_state,
    is_numeric_conflict,
    is_pin_conflict,
    is_geographic_conflict,
    evaluate_hard_veto
)

from precision_sieve_er.src.blocking import (
    SparseBM25Retriever,
    PrecisionSieveBlocker
)

from precision_sieve_er.src.features import (
    extract_pair_features,
    extract_candidate_features_dataframe
)

from precision_sieve_er.src.classifier import (
    AsymmetricPrecisionEnsemble
)

from precision_sieve_er.src.evaluation import (
    compute_entity_f_beta,
    evaluate_macro_f05,
    optimize_thresholds_grid
)

from precision_sieve_er.src.assignment import (
    apply_assignment_rules
)

from precision_sieve_er.src.pipeline import (
    run_precision_sieve_inference
)

__all__ = [
    "clean_entity_field",
    "normalize_text",
    "transliterate_to_latin",
    "create_normalized_dataframe",
    "extract_numbers",
    "extract_pin_codes",
    "extract_street_numbers",
    "extract_us_state",
    "is_numeric_conflict",
    "is_pin_conflict",
    "is_geographic_conflict",
    "evaluate_hard_veto",
    "SparseBM25Retriever",
    "PrecisionSieveBlocker",
    "extract_pair_features",
    "extract_candidate_features_dataframe",
    "AsymmetricPrecisionEnsemble",
    "compute_entity_f_beta",
    "evaluate_macro_f05",
    "optimize_thresholds_grid",
    "apply_assignment_rules",
    "run_precision_sieve_inference",
]
