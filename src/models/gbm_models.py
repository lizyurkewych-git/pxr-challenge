"""
Gradient boosting model wrappers for the PXR challenge.

Models:
  - LGBMWrapper  : LightGBM regressor with early stopping
  - XGBWrapper   : XGBoost regressor with early stopping
  - GBMEnsemble  : Simple equal-weight average of LightGBM + XGBoost

All wrappers accept numpy feature matrices and 1D target arrays.
Feature matrix should come from FeaturePipeline.fit_transform().

Usage:
    from src.models.gbm_models import LGBMWrapper, XGBWrapper, GBMEnsemble

    lgbm = LGBMWrapper()
    lgbm.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    preds = lgbm.predict(X_test)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBMWrapper:
    """LightGBM regressor tuned for small/medium molecular property datasets.

    Key choices:
    - MAE objective: aligns with the RAE metric (which uses MAE in numerator)
    - Low learning rate + many estimators: better generalization
    - subsample + colsample: regularization for high-dimensional fingerprints
    """

    DEFAULT_PARAMS = {
        "objective": "regression_l1",  # MAE loss
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 100,
    ) -> "LGBMWrapper":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        callbacks = [lgb.log_evaluation(period=200)]

        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            sample_weight=sample_weight,
            callbacks=callbacks,
        )
        best_iter = getattr(self._model, "best_iteration_", self.params["n_estimators"])
        logger.info(f"LightGBM fitted. Best iteration: {best_iter}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X).astype(np.float64)

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        return self._model.booster_.feature_importance(importance_type=importance_type)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBWrapper:
    """XGBoost regressor with MAE objective and early stopping."""

    DEFAULT_PARAMS = {
        "objective": "reg:absoluteerror",  # MAE loss (XGBoost 2.0+)
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",  # fast histogram-based method
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 100,
    ) -> "XGBWrapper":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self._model = xgb.XGBRegressor(**self.params)

        fit_kwargs: dict = {"X": X_train, "y": y_train}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = 200
            self._model.set_params(early_stopping_rounds=early_stopping_rounds)

        self._model.fit(**fit_kwargs)
        logger.info(f"XGBoost fitted. Best iteration: {self._model.best_iteration}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X).astype(np.float64)


# ---------------------------------------------------------------------------
# Random Forest (diversity member for ensemble)
# ---------------------------------------------------------------------------

class RFWrapper:
    """Random Forest regressor — provides ensemble diversity due to different
    tree construction vs. gradient boosting."""

    DEFAULT_PARAMS = {
        "n_estimators": 2000,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RFWrapper":
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(**self.params)
        self._model.fit(X_train, y_train, sample_weight=sample_weight)
        logger.info("RandomForest fitted.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X).astype(np.float64)


# ---------------------------------------------------------------------------
# GBM Ensemble (LightGBM + XGBoost, equal weights)
# ---------------------------------------------------------------------------

class GBMEnsemble:
    """Simple equal-weight ensemble of LightGBM + XGBoost.

    Useful as a quick strong baseline before adding GNN models.
    """

    def __init__(self, lgbm_params: Optional[dict] = None, xgb_params: Optional[dict] = None):
        self.lgbm = LGBMWrapper(lgbm_params)
        self.xgb = XGBWrapper(xgb_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "GBMEnsemble":
        logger.info("Fitting LightGBM...")
        self.lgbm.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=sample_weight)
        logger.info("Fitting XGBoost...")
        self.xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        lgbm_pred = self.lgbm.predict(X)
        xgb_pred = self.xgb.predict(X)
        return (lgbm_pred + xgb_pred) / 2.0


# ---------------------------------------------------------------------------
# Optuna hyperparameter search (use sparingly — data quality > tuning)
# ---------------------------------------------------------------------------

def tune_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: int = 3600,  # seconds
) -> dict:
    """Optuna-based hyperparameter search for LightGBM.

    Returns the best params dict to pass into LGBMWrapper.

    Note: Per ExpansionRx lessons, hyperparameter optimization had minimal
    impact. Run this only after data/feature engineering is finalized.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna not installed. Run: pip install optuna")

    from src.evaluation.validate import rae

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "regression_l1",
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMWrapper(params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        val_pred = model.predict(X_val)
        return rae(y_val, val_pred, y_train_mean=float(y_train.mean()))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Best LGBM RAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200, 100)).astype(np.float32)
    y = rng.uniform(3, 8, 200)
    X_val = rng.uniform(0, 1, (50, 100)).astype(np.float32)
    y_val = rng.uniform(3, 8, 50)

    lgbm = LGBMWrapper()
    lgbm.fit(X[:150], y[:150], X_val=X[150:], y_val=y[150:])
    preds = lgbm.predict(X_val)
    print("LGBM preds:", preds[:5])
