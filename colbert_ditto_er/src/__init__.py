"""
ColBERT-Ditto Entity Resolution Package.
Late-Interaction Transformer Architecture for High-Precision Record Linkage.
"""

from colbert_ditto_er.src.normalization import normalize_text, transliterate_to_latin, clean_entity_field
from colbert_ditto_er.src.serialization import serialize_record, serialize_dataframe
from colbert_ditto_er.src.blocking import SparseBlockingEngine
from colbert_ditto_er.src.colbert_model import ColBERTTokenEncoder, compute_maxsim_score
from colbert_ditto_er.src.evaluation import compute_entity_f_beta, evaluate_macro_f05, optimize_thresholds_grid
from colbert_ditto_er.src.assignment import apply_assignment_rules
from colbert_ditto_er.src.pipeline import run_streaming_colbert_inference

__all__ = [
    "normalize_text",
    "transliterate_to_latin",
    "clean_entity_field",
    "serialize_record",
    "serialize_dataframe",
    "SparseBlockingEngine",
    "ColBERTTokenEncoder",
    "compute_maxsim_score",
    "compute_entity_f_beta",
    "evaluate_macro_f05",
    "optimize_thresholds_grid",
    "apply_assignment_rules",
    "run_streaming_colbert_inference",
]
