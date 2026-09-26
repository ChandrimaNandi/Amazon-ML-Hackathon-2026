"""
Precision-Oriented Asymmetric LightGBM Entity Matcher Module.
Optimizes for Macro F0.5 by penalizing false positive merges with asymmetric sample weights.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from fast_hybrid_er.configs.default_config import MODEL_PARAMS, FP_PENALTY_WEIGHT, RESULTS_DIR
from fast_hybrid_er.src.features import FEATURE_COLUMNS

logger = logging.getLogger("FastHybridER.Classifier")


class AsymmetricEntityRanker:
    """LightGBM Binary Classifier trained with precision-heavy asymmetric weights."""
    def __init__(self, params: Optional[Dict[str, Any]] = None, fp_penalty: float = FP_PENALTY_WEIGHT):
        self.params = params or MODEL_PARAMS.copy()
        self.fp_penalty = fp_penalty
        self.model = None
        self.feature_names = FEATURE_COLUMNS.copy()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Trains LightGBM model with asymmetric sample weights:
        Negatives are weighted with self.fp_penalty to penalize false positives.
        """
        if not HAS_LGB:
            raise RuntimeError("lightgbm package is required for AsymmetricEntityRanker.")

        logger.info(f"[TRAINING] Training Asymmetric LightGBM on {len(X_train):,} samples with FP penalty {self.fp_penalty}...")

        # Feature matrix
        X_tr = X_train[self.feature_names].values
        # Sample weights: positive=1.0, negative=fp_penalty
        sample_weights = np.where(y_train == 1, 1.0, self.fp_penalty).astype(np.float32)

        train_data = lgb.Dataset(X_tr, label=y_train, weight=sample_weights, feature_name=self.feature_names)

        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            X_v = X_val[self.feature_names].values
            val_weights = np.where(y_val == 1, 1.0, self.fp_penalty).astype(np.float32)
            val_data = lgb.Dataset(X_v, label=y_val, weight=val_weights, feature_name=self.feature_names, reference=train_data)
            valid_sets.append(val_data)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            callbacks=[lgb.log_evaluation(period=100)]
        )
        logger.info("[TRAINING] Model training completed successfully.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability of match for each candidate pair."""
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train() or load() first.")
        X_mat = X[self.feature_names].values
        return self.model.predict(X_mat)

    def save(self, path: Optional[Path] = None) -> Path:
        """Saves model artifact to disk."""
        save_path = path or (RESULTS_DIR / "hybrid_er_model.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"[MODEL] Saved model to {save_path}")
        return save_path

    def load(self, path: Optional[Path] = None):
        """Loads model artifact from disk."""
        load_path = path or (RESULTS_DIR / "hybrid_er_model.pkl")
        with open(load_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"[MODEL] Loaded model from {load_path}")
