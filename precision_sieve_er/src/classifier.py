"""
Pillar 2 & 4: Asymmetric 4x False-Positive Loss GBDT Ensemble (LightGBM + CatBoost / XGBoost).
Directly optimizes for Macro F0.5 precision dominance and protects singletons from false merges.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle

import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from precision_sieve_er.src.veto_gates import evaluate_hard_veto

logger = logging.getLogger("PrecisionSieve.Classifier")

EXCLUDED_COLS = {"query_id", "s1_id", "label"}


class AsymmetricPrecisionEnsemble:
    """
    Heterogeneous GBDT Ensemble trained with Asymmetric 4x False-Positive Penalty.
    Ensembles LightGBM + CatBoost (or XGBoost) to eliminate model variance and reach 99+ F0.5.
    """
    def __init__(
        self,
        fp_weight: float = 4.0,
        n_estimators: int = 500,
        learning_rate: float = 0.04
    ):
        self.fp_weight = fp_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_cols: List[str] = []
        
        # 1. Primary Model: LightGBM (Leaf-wise deep splits)
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=45,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Secondary Model: CatBoost (Oblivious trees) or XGBoost
        if HAS_CATBOOST:
            self.secondary_model = CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=6,
                random_seed=42,
                verbose=False
            )
            self.secondary_type = "catboost"
        elif HAS_XGBOOST:
            self.secondary_model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            self.secondary_type = "xgboost"
        else:
            self.secondary_model = None
            self.secondary_type = "none"

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        self.feature_cols = [c for c in X_df.columns if c not in EXCLUDED_COLS]
        X = X_df[self.feature_cols].values
        
        # Pillar 2: Asymmetric 4x False-Positive Loss
        # False merges are penalized 4x more heavily to protect singletons
        sample_weights = np.where(y == 1, 1.0, self.fp_weight)
        
        logger.info(f"[TRAINING] Fitting LightGBM on {len(X):,} candidate pairs (FP weight = {self.fp_weight}x)...")
        self.lgb_model.fit(X, y, sample_weight=sample_weights)
        
        if self.secondary_model is not None:
            logger.info(f"[TRAINING] Fitting {self.secondary_type.upper()} ensemble model...")
            self.secondary_model.fit(X, y, sample_weight=sample_weights)
        logger.info("[TRAINING] Ensemble training complete.")

    def predict_proba(
        self,
        feat_df: pd.DataFrame,
        apply_vetoes: bool = True,
        s1_lookup: Optional[Dict[str, Dict[str, str]]] = None,
        query_lookup: Optional[Dict[str, Dict[str, str]]] = None
    ) -> np.ndarray:
        """
        Predicts ensemble matching probabilities.
        Applies Pillar 1 Hard Veto Gates to force confirmed contradictions to 0.0.
        """
        X = feat_df[self.feature_cols].values
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        
        if self.secondary_model is not None:
            p_sec = self.secondary_model.predict_proba(X)[:, 1]
            p_final = 0.60 * p_lgb + 0.40 * p_sec
        else:
            p_final = p_lgb
            
        # Apply Pillar 1 Hard Veto Gates
        if apply_vetoes and s1_lookup is not None and query_lookup is not None:
            for idx, row in feat_df.iterrows():
                qid = row["query_id"]
                s1id = row["s1_id"]
                q_data = query_lookup.get(qid, {})
                s_data = s1_lookup.get(s1id, {})
                
                if evaluate_hard_veto(
                    q_addr=q_data.get("addr_norm", ""),
                    s_addr=s_data.get("addr_norm", ""),
                    q_country=q_data.get("country", ""),
                    s_country=s_data.get("country", "")
                ):
                    p_final[idx] = 0.0

        return p_final

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "AsymmetricPrecisionEnsemble":
        with open(filepath, "rb") as f:
            return pickle.load(f)
