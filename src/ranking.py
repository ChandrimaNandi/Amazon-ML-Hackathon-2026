"""
Ranking & Matching Model Module using LightGBM with Multi-Epoch Iterative HNM.
"""

import time
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.config import MODEL_PARAMS
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class EntityMatcherModel:
    """
    LightGBM-based Candidate Matcher.
    Estimates P(candidate_pair is true match).
    Supports multi-epoch iterative training with alternate-round Hard Negative Mining (HNM).
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

    def fit_iterative_hnm(
        self,
        candidate_feat_df: pd.DataFrame,
        num_epochs: int = 5,
        hnm_every: int = 2,
        initial_threshold: float = 0.25,
        val_df: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Trains model over num_epochs, mining hard negatives on alternate epochs.
        """
        logger.info("=" * 60)
        logger.info(f"STARTING MULTI-EPOCH ITERATIVE TRAINING (Epochs: {num_epochs}, HNM Every: {hnm_every})")
        logger.info("=" * 60)
        
        epoch_stats = []
        current_train_df = candidate_feat_df.copy()
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n--- EPOCH {epoch}/{num_epochs} ---")
            
            # 1. Fit model on current training data
            self.fit(current_train_df, val_df=val_df)
            
            # 2. Check metrics on candidate set
            probs = self.predict_proba(candidate_feat_df)
            candidate_feat_df["pred_score"] = probs
            
            num_pos = (candidate_feat_df["is_match"] == 1).sum()
            high_conf_fp = ((candidate_feat_df["is_match"] == 0) & (candidate_feat_df["pred_score"] >= 0.50)).sum()
            
            logger.info(f"Epoch {epoch} Candidate Set: Total={len(candidate_feat_df):,}, Positives={num_pos:,}, False Positives (score>=0.5)={high_conf_fp:,}")
            
            # 3. Alternate Epoch Hard Negative Mining
            mined_count = 0
            if epoch % hnm_every == 0 and epoch < num_epochs:
                # Dynamically tighten threshold on later HNM rounds
                thresh = max(0.15, initial_threshold - (epoch * 0.02))
                hard_negs = mine_hard_negatives(self, candidate_feat_df, threshold=thresh)
                mined_count = len(hard_negs)
                
                if mined_count > 0:
                    logger.info(f"Augmenting training set with {mined_count:,} mined hard negatives (score >= {thresh:.2f})...")
                    # Duplicate hard negatives to give them higher learning weight
                    current_train_df = pd.concat([current_train_df, hard_negs, hard_negs], ignore_index=True)
                    logger.info(f"New Training Set Size: {len(current_train_df):,} pairs")
                    
            epoch_stats.append({
                "epoch": epoch,
                "train_size": len(current_train_df),
                "mined_hard_negatives": mined_count,
                "false_positives_gte_05": high_conf_fp
            })
            
        logger.info("=" * 60)
        logger.info("MULTI-EPOCH ITERATIVE HNM TRAINING COMPLETE.")
        logger.info("=" * 60)
        return epoch_stats

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

    def save_model(self, filepath: Path):
        """Saves model instance to pickle file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved EntityMatcherModel to {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> "EntityMatcherModel":
        """Loads model instance from pickle file."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded EntityMatcherModel from {filepath}")
        return model


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
