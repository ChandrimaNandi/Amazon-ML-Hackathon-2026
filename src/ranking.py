"""
Ranking & Matching Model Module using LightGBM.

Estimates P(candidate_pair is true match) using gradient-boosted decision trees.
Provides rigorous training with:
- Reproducible random seed
- Early stopping on genuinely unseen validation set
- Feature importance diagnostics
- Clean metrics recording (rows, positive ratio, feature count, training time, best score)
"""

import time
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.config import MODEL_PARAMS, RANDOM_SEED
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class EntityMatcherModel:
    """
    LightGBM-based Precision-Oriented Candidate Matcher.
    Estimates P(candidate_pair is true match).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else MODEL_PARAMS.copy()
        if "random_state" not in self.params:
            self.params["random_state"] = RANDOM_SEED
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: List[str] = [col for col in FEATURE_COLUMNS]
        self.training_summary: Dict[str, Any] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        early_stopping_rounds: int = 40
    ) -> Dict[str, Any]:
        """
        Fits LightGBM classifier on train_df with optional early stopping on val_df.
        
        Args:
            train_df: Training DataFrame containing feature columns and 'is_match'.
            val_df: Optional validation DataFrame.
            feature_cols: Optional subset of feature columns to use.
            early_stopping_rounds: Early stopping patience.
            
        Returns:
            summary: Dictionary of training diagnostics and metrics.
        """
        if feature_cols is not None:
            self.feature_names = [c for c in feature_cols if c in train_df.columns]
        else:
            self.feature_names = [c for c in FEATURE_COLUMNS if c in train_df.columns]
            
        X_train = train_df[self.feature_names]
        y_train = train_df["is_match"].astype(int)
        
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        pos_ratio = n_pos / max(len(y_train), 1)
        
        logger.info("=" * 60)
        logger.info(f"[MODEL TRAINING] Fitting LightGBM on {len(X_train):,} pairs ({n_pos:,} positives, {n_neg:,} negatives, {pos_ratio*100:.1f}% positive)")
        logger.info(f"[MODEL TRAINING] Features ({len(self.feature_names)}): {self.feature_names[:8]}...")
        logger.info("=" * 60)
        
        start_t = time.time()
        
        # Adjust scale_pos_weight or use balanced parameters if needed
        model_params = self.params.copy()
        # LightGBM Classifier with robust GPU/CPU fallback
        self.model = lgb.LGBMClassifier(**model_params)
        
        callbacks = []
        eval_set = None
        if val_df is not None and not val_df.empty:
            X_val = val_df[self.feature_names]
            y_val = val_df["is_match"].astype(int)
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            
        try:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=callbacks if callbacks else None
            )
        except Exception as e:
            if "gpu" in str(model_params.get("device", "")).lower() or "cuda" in str(e).lower() or "gpu" in str(e).lower():
                logger.warning(
                    f"[GPU FALLBACK: LightGBM GPU tree learner unavailable ({e}). "
                    f"Falling back to multi-threaded CPU OpenMP]"
                )
                model_params["device"] = "cpu"
                self.model = lgb.LGBMClassifier(**model_params)
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    callbacks=callbacks if callbacks else None
                )
            else:
                raise e
        
        elapsed = time.time() - start_t
        best_iter = getattr(self.model, "best_iteration_", self.model.n_estimators)
        
        val_score = None
        if val_df is not None and not val_df.empty:
            val_probs = self.predict_proba(val_df)
            val_loss = float(-np.mean(y_val * np.log(np.clip(val_probs, 1e-7, 1 - 1e-7)) +
                                      (1 - y_val) * np.log(np.clip(1 - val_probs, 1e-7, 1 - 1e-7))))
            val_score = round(val_loss, 4)
            logger.info(f"[MODEL TRAINING] Validation LogLoss: {val_loss:.4f} at best iteration {best_iter}")
            
        self.training_summary = {
            "training_rows": len(X_train),
            "positive_rows": n_pos,
            "negative_rows": n_neg,
            "positive_ratio": round(pos_ratio, 4),
            "num_features": len(self.feature_names),
            "training_time_seconds": round(elapsed, 2),
            "best_iteration": best_iter,
            "val_logloss": val_score,
        }
        
        logger.info(f"[MODEL TRAINING] LightGBM training finished in {elapsed:.2f}s (Best Iteration: {best_iter}).")
        return self.training_summary

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns 1D array of predicted match probabilities P(is_match=1)."""
        if self.model is None:
            raise ValueError("EntityMatcherModel is not fitted.")
        if df.empty:
            return np.array([], dtype=np.float32)
            
        X = df[self.feature_names]
        probs = self.model.predict_proba(X)[:, 1]
        return probs.astype(np.float32)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns DataFrame of feature importances sorted descending."""
        if self.model is None:
            return pd.DataFrame()
            
        imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp

    def save_model(self, filepath: Path):
        """Serializes model instance to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"[MODEL] Model successfully saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> "EntityMatcherModel":
        """Loads model instance from disk."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"[MODEL] Model loaded from {filepath}")
        return model
