"""
High-Speed Vectorized Feature Engineering Module for Fast Hybrid ER.
Combines Phonetic agreement, RapidFuzz string metrics, Dense GPU semantic scores,
and Open-Set Country signals into a compact tabular feature matrix.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging

try:
    import rapidfuzz
    from rapidfuzz.distance import Levenshtein, JaroWinkler
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from fast_hybrid_er.src.phonetics import soundex, metaphone_key, phonetic_token_jaccard
from fast_hybrid_er.src.dense_encoder import DualGPUSemanticScorer

logger = logging.getLogger("FastHybridER.Features")


FEATURE_COLUMNS = [
    # Exact Match Signals
    "exact_name_raw",
    "exact_name_normalized",
    "exact_addr_normalized",
    "exact_comb_normalized",
    # Name String Metrics
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_token_overlap",
    "name_char_len_ratio",
    # Address String Metrics
    "addr_levenshtein",
    "addr_jaro_winkler",
    "addr_token_sort",
    "addr_token_overlap",
    "addr_char_len_ratio",
    # Phonetic Signals
    "soundex_match",
    "metaphone_match",
    "phonetic_token_jaccard",
    # Dense GPU Semantic Similarity
    "dense_semantic_name",
    "dense_semantic_comb",
    # Multi-Channel Retrieval Flags
    "by_exact_name",
    "by_exact_address",
    "by_exact_combined",
    "by_phonetic",
    "by_lsh",
    "by_token",
    "channel_agreement",
    # Country Concordance (Open-set friendly)
    "country_exact_match",
    "country_mismatch",
    "country_missing",
]


class HybridFeatureExtractor:
    """Extracts deterministic hybrid features for candidate pairs."""
    def __init__(self, dense_scorer: Optional[DualGPUSemanticScorer] = None):
        self.dense_scorer = dense_scorer or DualGPUSemanticScorer()

    def extract_features(
        self,
        candidate_records: List[Dict[str, Any]],
        s1_lookup: Dict[str, Dict[str, Any]],
        query_lookup: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Extracts feature vectors for all candidate pairs in candidate_records.
        Uses fast dictionary lookups and vectorized batch operations.
        """
        n_pairs = len(candidate_records)
        if n_pairs == 0:
            return pd.DataFrame(columns=["query_id", "s1_id"] + FEATURE_COLUMNS)

        logger.info(f"[FEATURES] Extracting hybrid features for {n_pairs:,} candidate pairs...")

        # Pre-allocate feature arrays
        exact_name_raw = np.zeros(n_pairs, dtype=np.float32)
        exact_name_norm = np.zeros(n_pairs, dtype=np.float32)
        exact_addr_norm = np.zeros(n_pairs, dtype=np.float32)
        exact_comb_norm = np.zeros(n_pairs, dtype=np.float32)

        name_lev = np.zeros(n_pairs, dtype=np.float32)
        name_jw = np.zeros(n_pairs, dtype=np.float32)
        name_ts = np.zeros(n_pairs, dtype=np.float32)
        name_tset = np.zeros(n_pairs, dtype=np.float32)
        name_to = np.zeros(n_pairs, dtype=np.float32)
        name_clr = np.zeros(n_pairs, dtype=np.float32)

        addr_lev = np.zeros(n_pairs, dtype=np.float32)
        addr_jw = np.zeros(n_pairs, dtype=np.float32)
        addr_ts = np.zeros(n_pairs, dtype=np.float32)
        addr_to = np.zeros(n_pairs, dtype=np.float32)
        addr_clr = np.zeros(n_pairs, dtype=np.float32)

        sndx_m = np.zeros(n_pairs, dtype=np.float32)
        meta_m = np.zeros(n_pairs, dtype=np.float32)
        phon_jac = np.zeros(n_pairs, dtype=np.float32)

        by_en = np.zeros(n_pairs, dtype=np.float32)
        by_ea = np.zeros(n_pairs, dtype=np.float32)
        by_ec = np.zeros(n_pairs, dtype=np.float32)
        by_ph = np.zeros(n_pairs, dtype=np.float32)
        by_ls = np.zeros(n_pairs, dtype=np.float32)
        by_tk = np.zeros(n_pairs, dtype=np.float32)
        ch_agr = np.zeros(n_pairs, dtype=np.float32)

        cntry_match = np.zeros(n_pairs, dtype=np.float32)
        cntry_mismatch = np.zeros(n_pairs, dtype=np.float32)
        cntry_missing = np.zeros(n_pairs, dtype=np.float32)

        q_names, s_names = [], []
        q_combs, s_combs = [], []
        pair_qids, pair_s1ids = [], []

        for i, c in enumerate(candidate_records):
            qid = c["query_id"]
            s1id = c["s1_id"]
            pair_qids.append(qid)
            pair_s1ids.append(s1id)

            q_rec = query_lookup.get(qid, {})
            s_rec = s1_lookup.get(s1id, {})

            q_nm_raw = q_rec.get("business_name", "")
            s_nm_raw = s_rec.get("business_name", "")
            q_nm = q_rec.get("name_transliterated", "") or q_rec.get("name_normalized", "")
            s_nm = s_rec.get("name_transliterated", "") or s_rec.get("name_normalized", "")
            q_ad = q_rec.get("address_transliterated", "") or q_rec.get("address_normalized", "")
            s_ad = s_rec.get("address_transliterated", "") or s_rec.get("address_normalized", "")
            q_cb = q_rec.get("combined_transliterated", "") or q_rec.get("combined_normalized", "")
            s_cb = s_rec.get("combined_transliterated", "") or s_rec.get("combined_normalized", "")

            q_names.append(q_nm)
            s_names.append(s_nm)
            q_combs.append(q_cb)
            s_combs.append(s_cb)

            # Exact Match
            exact_name_raw[i] = 1.0 if q_nm_raw and q_nm_raw == s_nm_raw else 0.0
            exact_name_norm[i] = 1.0 if q_nm and q_nm == s_nm else 0.0
            exact_addr_norm[i] = 1.0 if q_ad and q_ad == s_ad else 0.0
            exact_comb_norm[i] = 1.0 if q_cb and q_cb == s_cb else 0.0

            # Name distances
            if HAS_RAPIDFUZZ:
                name_lev[i] = Levenshtein.normalized_similarity(q_nm, s_nm)
                name_jw[i] = JaroWinkler.similarity(q_nm, s_nm)
                name_ts[i] = fuzz.token_sort_ratio(q_nm, s_nm) / 100.0
                name_tset[i] = fuzz.token_set_ratio(q_nm, s_nm) / 100.0
            else:
                name_lev[i] = 1.0 if q_nm == s_nm else 0.0
                name_jw[i] = name_lev[i]
                name_ts[i] = name_lev[i]
                name_tset[i] = name_lev[i]

            q_tokens = set(q_nm.split())
            s_tokens = set(s_nm.split())
            u_tokens = len(q_tokens | s_tokens)
            name_to[i] = len(q_tokens & s_tokens) / u_tokens if u_tokens > 0 else 0.0
            max_len = max(len(q_nm), len(s_nm))
            name_clr[i] = min(len(q_nm), len(s_nm)) / max_len if max_len > 0 else 1.0

            # Address distances
            if HAS_RAPIDFUZZ:
                addr_lev[i] = Levenshtein.normalized_similarity(q_ad, s_ad)
                addr_jw[i] = JaroWinkler.similarity(q_ad, s_ad)
                addr_ts[i] = fuzz.token_sort_ratio(q_ad, s_ad) / 100.0
            else:
                addr_lev[i] = 1.0 if q_ad == s_ad else 0.0
                addr_jw[i] = addr_lev[i]
                addr_ts[i] = addr_lev[i]

            q_atok = set(q_ad.split())
            s_atok = set(s_ad.split())
            u_atok = len(q_atok | s_atok)
            addr_to[i] = len(q_atok & s_atok) / u_atok if u_atok > 0 else 0.0
            max_alen = max(len(q_ad), len(s_ad))
            addr_clr[i] = min(len(q_ad), len(s_ad)) / max_alen if max_alen > 0 else 1.0

            # Phonetic signals
            q_first = q_nm.split()[0] if q_nm.split() else ""
            s_first = s_nm.split()[0] if s_nm.split() else ""
            sndx_m[i] = 1.0 if (q_first and s_first and soundex(q_first) == soundex(s_first)) else 0.0
            meta_m[i] = 1.0 if (q_first and s_first and metaphone_key(q_first) == metaphone_key(s_first)) else 0.0
            phon_jac[i] = phonetic_token_jaccard(q_nm, s_nm)

            # Channel flags
            by_en[i] = c.get("by_exact_name", 0)
            by_ea[i] = c.get("by_exact_address", 0)
            by_ec[i] = c.get("by_exact_combined", 0)
            by_ph[i] = c.get("by_phonetic", 0)
            by_ls[i] = c.get("by_lsh", 0)
            by_tk[i] = c.get("by_token", 0)
            ch_agr[i] = c.get("channel_agreement", 0)

            # Country signals
            q_c = str(q_rec.get("country", "")).strip().casefold()
            s_c = str(s_rec.get("country", "")).strip().casefold()
            if not q_c or not s_c:
                cntry_missing[i] = 1.0
            elif q_c == s_c:
                cntry_match[i] = 1.0
            else:
                cntry_mismatch[i] = 1.0

        # Dense GPU scoring on dual T4 in FP16
        dense_name_scores = self.dense_scorer.score_pairs(q_names, s_names)
        dense_comb_scores = self.dense_scorer.score_pairs(q_combs, s_combs)

        feat_dict = {
            "query_id": pair_qids,
            "s1_id": pair_s1ids,
            "exact_name_raw": exact_name_raw,
            "exact_name_normalized": exact_name_norm,
            "exact_addr_normalized": exact_addr_norm,
            "exact_comb_normalized": exact_comb_norm,
            "name_levenshtein": name_lev,
            "name_jaro_winkler": name_jw,
            "name_token_sort": name_ts,
            "name_token_set": name_tset,
            "name_token_overlap": name_to,
            "name_char_len_ratio": name_clr,
            "addr_levenshtein": addr_lev,
            "addr_jaro_winkler": addr_jw,
            "addr_token_sort": addr_ts,
            "addr_token_overlap": addr_to,
            "addr_char_len_ratio": addr_clr,
            "soundex_match": sndx_m,
            "metaphone_match": meta_m,
            "phonetic_token_jaccard": phon_jac,
            "dense_semantic_name": dense_name_scores,
            "dense_semantic_comb": dense_comb_scores,
            "by_exact_name": by_en,
            "by_exact_address": by_ea,
            "by_exact_combined": by_ec,
            "by_phonetic": by_ph,
            "by_lsh": by_ls,
            "by_token": by_tk,
            "channel_agreement": ch_agr,
            "country_exact_match": cntry_match,
            "country_mismatch": cntry_mismatch,
            "country_missing": cntry_missing,
        }

        return pd.DataFrame(feat_dict)
