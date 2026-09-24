"""
Ranking & Matching Model Module using LightGBM.
"""

import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.config import MODEL_PARAMS
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class EntityMatcherModel:
    """
    LightGBM-based Candidate Matcher.
    Estimates P(candidate_pair is true match).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else MODEL_PARAMS.copy()
        self.model: lgb.LGBMClassifier = None
        self.feature_names: List[str] = FEATURE_COLUMNS

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        val_df: Optional[pd.DataFrame] = None
    ):
        if feature_cols is not None:
            self.feature_names = feature_cols
            
        X_train = train_df[self.feature_names]
        y_train = train_df["is_match"]
        
        logger.info(f"Training LightGBM model on {len(X_train):,} pairs ({y_train.sum():,} positives)...")
        start_t = time.time()
        
        self.model = lgb.LGBMClassifier(**self.params)
        
        eval_set = None
        if val_df is not None:
            X_val = val_df[self.feature_names]
            y_val = val_df["is_match"]
            eval_set = [(X_val, y_val)]
            
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None
        )
        
        elapsed = time.time() - start_t
        logger.info(f"LightGBM trained in {elapsed:.2f}s.")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns array of predicted match probabilities."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        X = df[self.feature_names]
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns DataFrame of feature importances."""
        if self.model is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        return imp


def mine_hard_negatives(
    model: EntityMatcherModel,
    feat_df: pd.DataFrame,
    threshold: float = 0.25
) -> pd.DataFrame:
    """
    Identifies false positives with model score >= threshold as hard negatives.
    """
    probs = model.predict_proba(feat_df)
    feat_df = feat_df.copy()
    feat_df["pred_score"] = probs
    
    hard_negs = feat_df[(feat_df["is_match"] == 0) & (feat_df["pred_score"] >= threshold)]
    logger.info(f"Mined {len(hard_negs):,} hard negatives (score >= {threshold})")
    return hard_negs
