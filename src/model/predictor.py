"""
Modelo predictivo de subida de precio.

Entrena un clasificador LightGBM para predecir si una moneda subirá
más de un 2 % en las próximas 24 horas.  Expone métodos para:

- Preparar datos de entrenamiento (labeling).
- Entrenar y guardar el modelo.
- Cargar un modelo guardado.
- Predecir probabilidades sobre datos nuevos.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
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

class PricePredictor:
    """Clasificador binario para predicción de subida de precio."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: lgb.LGBMClassifier | None = None
        self._feature_cols = get_feature_columns()

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, klines_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el modelo con datos de múltiples monedas.

        Devuelve métricas de evaluación.
        """
        logger.info("Preparando datos de entrenamiento con %d monedas", len(klines_by_symbol))

        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series] = []

        for symbol, df in klines_by_symbol.items():
            featured = compute_features(df)
            labels = create_labels(featured, self._config.target_pct_change)

            # Eliminar filas sin label (últimas *horizon* velas)
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

        # Time series split para evaluación
        tscv = TimeSeriesSplit(n_splits=3)
        auc_scores: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(
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
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            auc_scores.append(auc)
            logger.info("Fold %d — AUC: %.4f", fold, auc)

        # Entrenamiento final con todos los datos
        self._model = lgb.LGBMClassifier(
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
        self._model.fit(X, y)

        # Reporte
        y_pred = self._model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        logger.info("Reporte final:\n%s", classification_report(y, y_pred))

        mean_auc = float(np.mean(auc_scores))
        metrics = {
            "mean_auc": mean_auc,
            "accuracy": report["accuracy"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
            "samples": len(X),
            "positive_rate": float(y.mean()),
        }
        logger.info("AUC medio CV: %.4f", mean_auc)
        return metrics

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, klines_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Devuelve {symbol: probabilidad_subida} para cada moneda.

        Solo incluye la última fila de features de cada moneda
        (= estado actual del mercado).
        """
        if self._model is None:
            raise RuntimeError("El modelo no está entrenado ni cargado.")

        predictions: dict[str, float] = {}
        for symbol, df in klines_by_symbol.items():
            try:
                featured = compute_features(df)
                if featured.empty:
                    continue
                last_row = featured[self._feature_cols].iloc[[-1]]
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
        """Guarda el modelo entrenado en disco."""
        save_path = path or MODEL_FILE
        if self._model is None:
            raise RuntimeError("No hay modelo para guardar.")
        joblib.dump(self._model, save_path)
        logger.info("Modelo guardado en %s", save_path)
        return save_path

    def load(self, path: Path | None = None) -> None:
        """Carga un modelo guardado desde disco."""
        load_path = path or MODEL_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {load_path}")
        self._model = joblib.load(load_path)
        logger.info("Modelo cargado desde %s", load_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def feature_importance(self) -> pd.DataFrame:
        """Devuelve importancia de cada feature (útil para debugging)."""
        if self._model is None:
            raise RuntimeError("Modelo no entrenado.")
        importance = self._model.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
