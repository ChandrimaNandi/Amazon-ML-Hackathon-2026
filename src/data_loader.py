"""
Data Loader Module for Business Entity Resolution.
Provides robust reading of TSV files and ground truth parsing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_source_tsv(path: Path) -> pd.DataFrame:
    """
    Loads a source TSV file (entity_id, business_name, business_address, country).
    Fills missing string values with empty string.
    """
    logger.info(f"Loading TSV file from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at {path}")
    
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )
    
    expected_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {path}, got {df.columns.tolist()}")
    
    # Clean whitespace and handle NaN/empty
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df["business_name"] = df["business_name"].astype(str).str.strip()
    df["business_address"] = df["business_address"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    
    logger.info(f"Loaded {len(df):,} rows from {path.name}")
    return df


def load_ground_truth(path: Path) -> Tuple[pd.DataFrame, Dict[str, Set[str]], Dict[str, str]]:
    """
    Loads train_ground_truth.tsv.
    Returns:
        gt_df: DataFrame with source1_entity_id and matched_entity_ids
        s1_to_matches: Dict mapping S1 ID -> set of matched S2/S3 IDs
        match_to_s1: Dict mapping S2/S3 ID -> S1 ID
    """
    logger.info(f"Loading Ground Truth from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at {path}")

    gt_df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )
    
    s1_to_matches: Dict[str, Set[str]] = {}
    match_to_s1: Dict[str, str] = {}
    
    s1_col = gt_df["source1_entity_id"].astype(str).str.strip()
    match_col = gt_df["matched_entity_ids"].astype(str).str.strip()
    
    for s1_id, matches_str in zip(s1_col, match_col):
        if matches_str:
            matched_set = set(m.strip() for m in matches_str.split(",") if m.strip())
        else:
            matched_set = set()
        
        s1_to_matches[s1_id] = matched_set
        for m in matched_set:
            match_to_s1[m] = s1_id
            
    logger.info(f"Loaded {len(s1_to_matches):,} ground truth S1 entities ({len(match_to_s1):,} total match mappings)")
    return gt_df, s1_to_matches, match_to_s1
