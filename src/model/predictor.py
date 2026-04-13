"""
Modelo predictivo de subida de precio.

Entrena un ensemble de 3 clasificadores (LightGBM, XGBoost, Random Forest)
que votan para predecir si una moneda subira mas de un X% en las proximas
N horas (configurable).  La probabilidad final es la media de los 3 modelos.

Expone metodos para:
- Preparar datos de entrenamiento (labeling).
- Entrenar y guardar el ensemble.
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
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import MODELS_DIR, ModelConfig
from src.data.features import compute_features, get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_FILE = MODELS_DIR / "predictor.joblib"


# ------------------------------------------------------------------
# Labeling
# ------------------------------------------------------------------

def create_labels(df: pd.DataFrame, target_pct: float, horizon: int = 24) -> pd.Series:
    """Genera etiquetas binarias: 1 si el precio sube >= *target_pct* en *horizon* velas."""
    future_close = df["close"].shift(-horizon)
    pct_change = (future_close - df["close"]) / df["close"]
    return (pct_change >= target_pct).astype(int)


# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

def _build_ensemble() -> VotingClassifier:
    """Construye un VotingClassifier con LightGBM, XGBoost y Random Forest."""
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # se ajusta dinamicamente en train()
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("lgbm", lgbm), ("xgb", xgb_clf), ("rf", rf)],
        voting="soft",  # promedia probabilidades en vez de votos duros
    )


class PricePredictor:
    """Ensemble de 3 modelos (LightGBM + XGBoost + Random Forest) para
    prediccion de subida de precio.  Usa voting suave (promedio de
    probabilidades)."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: VotingClassifier | None = None
        self._feature_cols = get_feature_columns(include_btc=True)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, klines_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el modelo con datos de multiples monedas.

        Extrae BTCUSDT del dict para usarlo como features de mercado.
        Devuelve metricas de evaluacion.
        """
        logger.info("Preparando datos de entrenamiento con %d monedas", len(klines_by_symbol))

        # Extraer BTC como proxy del mercado
        btc_df = klines_by_symbol.get("BTCUSDT")
        if btc_df is not None:
            logger.info("BTCUSDT disponible como proxy de mercado (%d filas)", len(btc_df))
        else:
            logger.warning("BTCUSDT no encontrado — features de BTC no disponibles")

        horizon = self._config.target_horizon_hours
        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series] = []

        for symbol, df in klines_by_symbol.items():
            featured = compute_features(df, btc_df=btc_df)
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

        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)

        logger.info(
            "Datos: %d muestras | positivas=%.1f%%",
            len(X),
            100 * y.mean(),
        )

        # Ajustar scale_pos_weight de XGBoost al ratio real
        neg_count = int((y == 0).sum())
        pos_count = int((y == 1).sum())
        spw = neg_count / max(pos_count, 1)

        # Time series split para evaluación
        tscv = TimeSeriesSplit(n_splits=3)
        auc_scores: list[float] = []
        per_model_auc: dict[str, list[float]] = {
            "lgbm": [], "xgb": [], "rf": [],
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            ensemble = _build_ensemble()
            # Ajustar XGBoost scale_pos_weight
            ensemble.named_estimators["xgb"].set_params(scale_pos_weight=spw)
            ensemble.fit(X_train, y_train)

            # AUC del ensemble
            y_prob = ensemble.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            auc_scores.append(auc)
            logger.info("Fold %d — Ensemble AUC: %.4f", fold, auc)

            # AUC por modelo individual
            for name, estimator in ensemble.named_estimators_.items():
                y_prob_i = estimator.predict_proba(X_val)[:, 1]
                auc_i = roc_auc_score(y_val, y_prob_i)
                per_model_auc[name].append(auc_i)
                logger.info("  Fold %d — %s AUC: %.4f", fold, name, auc_i)

        # Log de AUC medio por modelo
        for name, scores in per_model_auc.items():
            logger.info("%s AUC medio: %.4f", name, np.mean(scores))

        # Entrenamiento final con todos los datos
        self._model = _build_ensemble()
        self._model.named_estimators["xgb"].set_params(scale_pos_weight=spw)
        self._model.fit(X, y)

        # Reporte
        y_pred = self._model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        logger.info("Reporte final (ensemble):\n%s", classification_report(y, y_pred))

        mean_auc = float(np.mean(auc_scores))
        metrics: dict[str, Any] = {
            "mean_auc": mean_auc,
            "accuracy": report["accuracy"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
            "samples": len(X),
            "positive_rate": float(y.mean()),
        }
        # Incluir AUC por modelo individual
        for name, scores in per_model_auc.items():
            metrics[f"auc_{name}"] = float(np.mean(scores))

        logger.info("Ensemble AUC medio CV: %.4f", mean_auc)
        return metrics

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, klines_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Devuelve {symbol: probabilidad_subida} para cada moneda.

        Extrae BTCUSDT del dict para features de mercado.
        Solo incluye la ultima fila de features de cada moneda
        (= estado actual del mercado).
        """
        if self._model is None:
            raise RuntimeError("El modelo no esta entrenado ni cargado.")

        btc_df = klines_by_symbol.get("BTCUSDT")

        predictions: dict[str, float] = {}
        for symbol, df in klines_by_symbol.items():
            try:
                featured = compute_features(df, btc_df=btc_df)
                if featured.empty:
                    continue
                last_row = featured[self._feature_cols].iloc[[-1]]
                # predict_proba del VotingClassifier promedia las probabilidades
                prob = self._model.predict_proba(last_row)[0, 1]
                predictions[symbol] = float(prob)
            except Exception as exc:
                logger.warning("Error prediciendo %s: %s", symbol, exc)

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
        """Guarda el ensemble entrenado en disco."""
        save_path = path or MODEL_FILE
        if self._model is None:
            raise RuntimeError("No hay modelo para guardar.")
        joblib.dump(self._model, save_path)
        logger.info("Ensemble guardado en %s", save_path)
        return save_path

    def load(self, path: Path | None = None) -> None:
        """Carga un ensemble guardado desde disco."""
        load_path = path or MODEL_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {load_path}")
        self._model = joblib.load(load_path)
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

        importances: list[np.ndarray] = []
        for _name, estimator in self._model.named_estimators_.items():
            imp = np.array(estimator.feature_importances_, dtype=float)
            total = imp.sum()
            if total > 0:
                imp = imp / total  # normalizar a [0, 1]
            importances.append(imp)

        avg_importance = np.mean(importances, axis=0)
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": avg_importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
