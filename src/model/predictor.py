"""Modelo predictivo de subida de precio.

Entrena un ensemble voting de 3 clasificadores (LightGBM, XGBoost,
Random Forest) con voto suave (soft voting).  Los hiperparametros se
optimizan automaticamente con Optuna y las features se filtran
eliminando las de baja importancia.

Expone metodos para:
- Preparar datos de entrenamiento (labeling).
- Optimizar hiperparametros (Optuna).
- Seleccionar features automaticamente.
- Entrenar y guardar el voting ensemble.
- Cargar un ensemble guardado.
- Predecir probabilidades sobre datos nuevos.

Features incluyen indicadores tecnicos, MFI, soporte/resistencia, lags,
features temporales y features de BTC como proxy del mercado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import _QUOTE, MODELS_DIR, ModelConfig
from src.data.features import compute_features, compute_market_features, get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_FILE = MODELS_DIR / "predictor.joblib"

# Numero de trials de Optuna (equilibrio velocidad/calidad)
OPTUNA_N_TRIALS = 30

# Umbral minimo de importancia para conservar una feature (percentil)
FEATURE_IMPORTANCE_THRESHOLD_PCTILE = 10


# ------------------------------------------------------------------
# Labeling
# ------------------------------------------------------------------

def create_labels(df: pd.DataFrame, target_pct: float, horizon: int = 24) -> pd.Series:
    """Genera etiquetas binarias: 1 si el precio sube >= *target_pct* en *horizon* velas."""
    future_close = df["close"].shift(-horizon)
    pct_change = (future_close - df["close"]) / df["close"]
    return (pct_change >= target_pct).astype(int)


# ------------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------------

def _optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Funcion objetivo de Optuna: entrena un LightGBM con hiperparametros
    sugeridos y devuelve el AUC en validacion."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    model = lgb.LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)


def _run_optuna(
    X: pd.DataFrame, y: pd.Series, n_trials: int = OPTUNA_N_TRIALS,
) -> dict[str, Any]:
    """Ejecuta Optuna para encontrar los mejores hiperparametros.

    Usa el ultimo fold de TimeSeriesSplit como validacion para mantener
    el orden temporal.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]  # ultimo fold

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    logger.info(
        "Optuna completado: mejor AUC=%.4f en %d trials",
        study.best_value, n_trials,
    )
    logger.info("Mejores hiperparametros: %s", study.best_params)
    return study.best_params


# ------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------

def _select_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    lgbm_params: dict[str, Any],
) -> list[str]:
    """Elimina features con importancia por debajo del percentil umbral.

    Entrena un LightGBM rapido con los hiperparametros de Optuna y
    mira que features aportan menos.  Devuelve la lista filtrada.
    """
    model = lgb.LGBMClassifier(
        **lgbm_params,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X[feature_cols], y)

    importances = model.feature_importances_
    threshold = np.percentile(importances, FEATURE_IMPORTANCE_THRESHOLD_PCTILE)

    selected = [
        col for col, imp in zip(feature_cols, importances)
        if imp > threshold
    ]

    removed = set(feature_cols) - set(selected)
    if removed:
        logger.info(
            "Feature selection: eliminadas %d features de baja importancia: %s",
            len(removed), sorted(removed),
        )
    logger.info(
        "Features seleccionadas: %d de %d originales",
        len(selected), len(feature_cols),
    )
    return selected


# ------------------------------------------------------------------
# Voting ensemble builder
# ------------------------------------------------------------------

def _build_voting(
    lgbm_params: dict[str, Any],
    spw: float,
) -> VotingClassifier:
    """Construye un VotingClassifier (soft) con LightGBM, XGBoost y RF.

    Cada modelo vota con sus probabilidades y se promedian.
    Los hiperparametros de Optuna se comparten entre modelos.
    """
    lgbm = lgb.LGBMClassifier(
        **lgbm_params,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=lgbm_params.get("n_estimators", 300),
        learning_rate=lgbm_params.get("learning_rate", 0.05),
        max_depth=lgbm_params.get("max_depth", 6),
        subsample=lgbm_params.get("subsample", 0.8),
        colsample_bytree=lgbm_params.get("colsample_bytree", 0.8),
        reg_alpha=lgbm_params.get("reg_alpha", 0.01),
        reg_lambda=lgbm_params.get("reg_lambda", 1.0),
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=min(lgbm_params.get("max_depth", 6) + 4, 14),
        min_samples_leaf=lgbm_params.get("min_child_samples", 20),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("lgbm", lgbm), ("xgb", xgb_clf), ("rf", rf)],
        voting="soft",
        n_jobs=-1,
    )


# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

class PricePredictor:
    """Voting ensemble (LightGBM + XGBoost + Random Forest)
    con optimizacion Optuna y feature selection automatica."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: VotingClassifier | None = None
        self._calibrator: LogisticRegression | None = None
        self._feature_cols = get_feature_columns(include_btc=True)
        self._selected_features: list[str] | None = None

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, klines_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el modelo con datos de multiples monedas.

        Pipeline completo:
        1. Preparar datos (features + labels).
        2. Optuna -- buscar mejores hiperparametros.
        3. Feature selection -- eliminar features ruidosas.
        4. Voting ensemble -- entrenar con los parametros optimos.
        5. Evaluar con TimeSeriesSplit.

        Extrae BTCUSDT del dict para usarlo como features de mercado.
        Devuelve metricas de evaluacion.
        """
        logger.info("Preparando datos de entrenamiento con %d monedas", len(klines_by_symbol))

        # Extraer BTC como proxy del mercado
        btc_symbol = f"BTC{_QUOTE}"
        btc_df = klines_by_symbol.get(btc_symbol)
        if btc_df is not None:
            logger.info("%s disponible como proxy de mercado (%d filas)", btc_symbol, len(btc_df))
        else:
            logger.warning("%s no encontrado -- features de BTC no disponibles", btc_symbol)

        # Calcular features de mercado global (cross-coin)
        market_df = compute_market_features(klines_by_symbol)
        if not market_df.empty:
            logger.info("Features de mercado global calculadas (%d filas)", len(market_df))
        else:
            logger.warning("No se pudieron calcular features de mercado global")

        horizon = self._config.target_horizon_hours
        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series] = []

        for symbol, df in klines_by_symbol.items():
            featured = compute_features(
                df, btc_df=btc_df,
                market_df=market_df if not market_df.empty else None,
            )
            labels = create_labels(
                featured, self._config.target_pct_change, horizon=horizon,
            )

            # Eliminar filas sin label (ultimas *horizon* velas)
            valid_mask = labels.notna()
            featured = featured.loc[valid_mask]
            labels = labels.loc[valid_mask]

            if len(featured) < 20:
                continue

            all_X.append(featured[self._feature_cols])
            all_y.append(labels)

        if not all_X:
            raise ValueError("No hay suficientes datos para entrenar el modelo.")

        # Concatenar preservando el indice temporal para evitar data leakage.
        # Al ordenar por timestamp, TimeSeriesSplit respetara la cronologia
        # real incluso mezclando datos de distintas monedas.
        X = pd.concat(all_X)  # preservar indice temporal
        y = pd.concat(all_y)
        sort_order = X.index.argsort()
        X = X.iloc[sort_order].reset_index(drop=True)
        y = y.iloc[sort_order].reset_index(drop=True)

        logger.info(
            "Datos: %d muestras | positivas=%.1f%%",
            len(X),
            100 * y.mean(),
        )

        # Ajustar scale_pos_weight de XGBoost al ratio real
        neg_count = int((y == 0).sum())
        pos_count = int((y == 1).sum())
        spw = neg_count / max(pos_count, 1)

        # ----------------------------------------------------------
        # Paso 1: Optuna -- buscar mejores hiperparametros
        # ----------------------------------------------------------
        logger.info(
            "Paso 1/4: Optimizando hiperparametros con Optuna (%d trials)...",
            OPTUNA_N_TRIALS,
        )
        best_params = _run_optuna(X, y)

        # ----------------------------------------------------------
        # Paso 2: Feature selection
        # ----------------------------------------------------------
        logger.info("Paso 2/4: Seleccionando features...")
        self._selected_features = _select_features(
            X, y, self._feature_cols, best_params,
        )
        X_sel = X[self._selected_features]

        # ----------------------------------------------------------
        # Paso 3: Evaluar con Time Series Split
        # ----------------------------------------------------------
        logger.info("Paso 3/4: Evaluando con TimeSeriesSplit...")
        tscv = TimeSeriesSplit(n_splits=3)
        auc_scores: list[float] = []
        per_model_auc: dict[str, list[float]] = {
            "lgbm": [], "xgb": [], "rf": [],
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sel)):
            X_train, X_val = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            ensemble = _build_voting(best_params, spw)
            ensemble.fit(X_train, y_train)

            # AUC del voting ensemble
            y_prob = ensemble.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            auc_scores.append(auc)
            logger.info("Fold %d -- Voting AUC: %.4f", fold, auc)

            # AUC por modelo individual
            for name, estimator in ensemble.named_estimators_.items():
                y_prob_i = estimator.predict_proba(X_val)[:, 1]
                auc_i = roc_auc_score(y_val, y_prob_i)
                per_model_auc[name].append(auc_i)
                logger.info("  Fold %d -- %s AUC: %.4f", fold, name, auc_i)

        # Log de AUC medio por modelo
        for name, scores in per_model_auc.items():
            logger.info("%s AUC medio: %.4f", name, np.mean(scores))

        # Guardar metricas del ultimo fold de CV (las mas honestas)
        last_fold_idx = len(auc_scores) - 1
        last_train_idx, last_val_idx = list(tscv.split(X_sel))[last_fold_idx]
        X_val_last = X_sel.iloc[last_val_idx]
        y_val_last = y.iloc[last_val_idx]

        # Evaluar el ultimo ensemble de CV sobre su validation set
        y_pred_cv = ensemble.predict(X_val_last)
        cv_report = classification_report(y_val_last, y_pred_cv, output_dict=True)
        logger.info(
            "Metricas HONESTAS (ultimo fold CV, datos no vistos):\n%s",
            classification_report(y_val_last, y_pred_cv),
        )

        # ----------------------------------------------------------
        # Paso 4: Entrenamiento final + calibración Platt Scaling
        # ----------------------------------------------------------
        logger.info("Paso 4/4: Entrenamiento final + calibración Platt Scaling...")
        self._model = _build_voting(best_params, spw)
        self._model.fit(X_sel, y)

        # Platt Scaling: ajustar LogisticRegression sobre probabilidades
        # de CV para mapear prob_raw -> prob_calibrada.
        # Recogemos probabilidades out-of-fold del paso 3.
        oof_probs = np.full(len(y), np.nan)
        for train_idx, val_idx in tscv.split(X_sel):
            X_tr, X_vl = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            tmp_ens = _build_voting(best_params, spw)
            tmp_ens.fit(X_tr, y_tr)
            oof_probs[val_idx] = tmp_ens.predict_proba(X_vl)[:, 1]

        valid_mask = ~np.isnan(oof_probs)
        self._calibrator = LogisticRegression(max_iter=1000)
        self._calibrator.fit(
            oof_probs[valid_mask].reshape(-1, 1),
            y.values[valid_mask],
        )
        # Log de la calibración
        raw_sample = np.array([0.05, 0.10, 0.15, 0.20, 0.25]).reshape(-1, 1)
        cal_sample = self._calibrator.predict_proba(raw_sample)[:, 1]
        logger.info("Calibración Platt Scaling ajustada:")
        for raw_val, cal_val in zip(raw_sample.ravel(), cal_sample):
            logger.info("  raw=%.2f -> calibrada=%.1f%%", raw_val, cal_val * 100)

        mean_auc = float(np.mean(auc_scores))
        # Usar metricas de CV (honestas), no de training
        cv_cls1 = cv_report.get("1", {})
        metrics: dict[str, Any] = {
            "mean_auc": mean_auc,
            "cv_accuracy": cv_report.get("accuracy", 0),
            "cv_precision_1": cv_cls1.get("precision", 0),
            "cv_recall_1": cv_cls1.get("recall", 0),
            "cv_f1_1": cv_cls1.get("f1-score", 0),
            "samples": len(X_sel),
            "positive_rate": float(y.mean()),
            "n_features_original": len(self._feature_cols),
            "n_features_selected": len(self._selected_features),
            "best_params": best_params,
        }
        # Incluir AUC por modelo individual
        for name, scores in per_model_auc.items():
            metrics[f"auc_{name}"] = float(np.mean(scores))

        logger.info("Voting AUC medio CV: %.4f", mean_auc)
        logger.info(
            "Features: %d seleccionadas de %d originales",
            len(self._selected_features), len(self._feature_cols),
        )
        return metrics

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, klines_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Devuelve {symbol: probabilidad_subida} para cada moneda.

        Extrae BTC del dict para features de mercado.
        Solo incluye la ultima fila de features de cada moneda
        (= estado actual del mercado).
        """
        if self._model is None:
            raise RuntimeError("El modelo no esta entrenado ni cargado.")

        btc_df = klines_by_symbol.get(f"BTC{_QUOTE}")

        # Calcular features de mercado global para predicción
        market_df = compute_market_features(klines_by_symbol)
        mkt = market_df if not market_df.empty else None

        # Usar features seleccionadas si estan disponibles
        feat_cols = self._selected_features or self._feature_cols

        predictions: dict[str, float] = {}
        for symbol, df in klines_by_symbol.items():
            try:
                featured = compute_features(df, btc_df=btc_df, market_df=mkt)
                if featured.empty:
                    continue
                last_row = featured[feat_cols].iloc[[-1]]
                raw_prob = self._model.predict_proba(last_row)[0, 1]
                if self._calibrator is not None:
                    prob = float(
                        self._calibrator.predict_proba(
                            np.array([[raw_prob]]),
                        )[0, 1]
                    )
                else:
                    prob = float(raw_prob)
                predictions[symbol] = prob
            except Exception as exc:
                logger.warning("Error prediciendo %s: %s", symbol, exc)

        # Log de distribucion de probabilidades para diagnostico
        if predictions:
            probs = sorted(predictions.values(), reverse=True)
            logger.info(
                "Distribucion de probabilidades: max=%.3f | p75=%.3f | "
                "mediana=%.3f | p25=%.3f | min=%.3f",
                probs[0],
                probs[len(probs) // 4] if len(probs) >= 4 else probs[0],
                probs[len(probs) // 2],
                probs[3 * len(probs) // 4] if len(probs) >= 4 else probs[-1],
                probs[-1],
            )
            # Top 5 monedas por probabilidad
            top5 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
            for sym, prob in top5:
                logger.info("  Top: %s = %.3f (%.1f%%)", sym, prob, prob * 100)

        return predictions

    def get_recommendations(
        self,
        predictions: dict[str, float],
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """Filtra y ordena las monedas que superan el umbral de confianza.

        Devuelve lista de (symbol, probability) ordenada de mayor a menor.
        """
        thresh = threshold or self._config.confidence_threshold
        above = [(sym, prob) for sym, prob in predictions.items() if prob >= thresh]
        above.sort(key=lambda x: x[1], reverse=True)
        return above

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Guarda el ensemble y la lista de features seleccionadas."""
        save_path = path or MODEL_FILE
        if self._model is None:
            raise RuntimeError("No hay modelo para guardar.")
        payload = {
            "model": self._model,
            "calibrator": self._calibrator,
            "selected_features": self._selected_features,
        }
        joblib.dump(payload, save_path)
        logger.info("Voting ensemble guardado en %s", save_path)
        return save_path

    def load(self, path: Path | None = None) -> None:
        """Carga un ensemble guardado desde disco."""
        load_path = path or MODEL_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"No se encontro el modelo en {load_path}")
        payload = joblib.load(load_path)
        if isinstance(payload, dict):
            self._model = payload["model"]
            self._calibrator = payload.get("calibrator")
            self._selected_features = payload.get("selected_features")
        else:
            # Compatibilidad con modelos guardados antes del stacking
            self._model = payload
            self._calibrator = None
            self._selected_features = None
        logger.info("Ensemble cargado desde %s", load_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def feature_importance(self) -> pd.DataFrame:
        """Devuelve importancia media de cada feature (promediada entre modelos).

        LightGBM y XGBoost exponen feature_importances_ directamente.
        Random Forest tambien.  Promediamos las tres (normalizadas).
        """
        if self._model is None:
            raise RuntimeError("Modelo no entrenado.")

        feat_cols = self._selected_features or self._feature_cols
        importances: list[np.ndarray] = []
        for _name, estimator in self._model.named_estimators_.items():
            imp = np.array(estimator.feature_importances_, dtype=float)
            total = imp.sum()
            if total > 0:
                imp = imp / total  # normalizar a [0, 1]
            importances.append(imp)

        avg_importance = np.mean(importances, axis=0)
        return (
            pd.DataFrame({"feature": feat_cols, "importance": avg_importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
