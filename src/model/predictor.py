"""Modelo predictivo de subida de precio.

Entrena un stacking ensemble de 4 clasificadores base (LightGBM, XGBoost,
Random Forest, ExtraTrees) con un meta-learner (LogisticRegression).
Los hiperparametros se optimizan automaticamente con Optuna y las
features se filtran eliminando las de baja importancia.

Expone metodos para:
- Preparar datos de entrenamiento (labeling realista con TP/SL).
- Optimizar hiperparametros (Optuna).
- Seleccionar features automaticamente.
- Entrenar stacking ensemble con meta-learner calibrado.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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

# Gap entre folds de CV para evitar data leakage (en horas).
PURGE_GAP_HOURS = 48


# ------------------------------------------------------------------
# Labeling
# ------------------------------------------------------------------

def create_labels(
    df: pd.DataFrame,
    target_pct: float,
    horizon: int = 48,
    stop_loss_pct: float | None = None,
) -> pd.Series:
    """Genera etiquetas binarias simulando la estrategia TP/SL real.

    Para cada vela, recorre las siguientes *horizon* velas y comprueba:
    - Si el ``high`` toca ``target_pct`` (TP) ANTES de que el ``low``
      toque ``stop_loss_pct`` (SL) -> label = 1.
    - Si el SL se toca primero, o no se toca nada -> label = 0.

    Nota: cuando una misma vela toca AMBOS niveles, se asume SL primero
    (enfoque conservador — dentro de una vela horaria no sabemos el
    orden real).  Esto introduce un sesgo leve hacia labels negativos.

    Si *stop_loss_pct* es None, solo comprueba si el high toca TP
    en algun momento de la ventana.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas ``close``, ``high``, ``low``.
    target_pct : float
        Porcentaje de subida para TP (ej. 0.03 = +3%).
    horizon : int
        Numero de velas a mirar hacia adelante.
    stop_loss_pct : float | None
        Porcentaje de bajada para SL (ej. 0.03 = -3%).
        Si es None, solo se comprueba TP.
    """
    n = len(df)
    labels = np.full(n, np.nan)
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values

    for i in range(n - 1):
        entry_price = close_arr[i]
        tp_price = entry_price * (1 + target_pct)
        sl_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct else 0.0
        end_idx = min(i + horizon + 1, n)

        hit_tp = False
        hit_sl = False
        for j in range(i + 1, end_idx):
            if sl_price > 0 and low_arr[j] <= sl_price:
                hit_sl = True
                break
            if high_arr[j] >= tp_price:
                hit_tp = True
                break

        if i + horizon < n:
            labels[i] = 1.0 if hit_tp and not hit_sl else 0.0

    return pd.Series(labels, index=df.index)


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
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
    timestamps: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    """Ejecuta Optuna para encontrar los mejores hiperparametros.

    Usa Purged TimeSeriesSplit con TODOS los folds (promedio de AUC)
    para evitar sobreajuste al periodo mas reciente.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    splits = _purged_ts_split(
        len(X), n_splits=3,
        timestamps=timestamps, gap_hours=PURGE_GAP_HOURS,
    )

    def objective(trial: optuna.Trial) -> float:
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
        fold_aucs: list[float] = []
        for train_idx, val_idx in splits:
            model = lgb.LGBMClassifier(
                **params, class_weight="balanced",
                random_state=42, verbose=-1,
            )
            model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y.iloc[val_idx], y_prob))
        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Optuna completado: mejor AUC medio=%.4f en %d trials",
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
    train_idx: np.ndarray | None = None,
) -> list[str]:
    """Elimina features con importancia por debajo del percentil umbral.

    Entrena un LightGBM rapido con los hiperparametros de Optuna y
    mira que features aportan menos.  Devuelve la lista filtrada.

    Si *train_idx* se proporciona, solo usa esas filas para entrenar
    (evita data leakage usando datos de validacion para seleccionar).
    """
    X_fit = X[feature_cols].iloc[train_idx] if train_idx is not None else X[feature_cols]
    y_fit = y.iloc[train_idx] if train_idx is not None else y

    model = lgb.LGBMClassifier(
        **lgbm_params,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_fit, y_fit)

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
# Base model builders
# ------------------------------------------------------------------

def _build_base_models(
    lgbm_params: dict[str, Any],
    spw: float,
) -> dict[str, Any]:
    """Construye los 4 modelos base para el stacking.

    Returns dict {name: estimator} sin entrenar.
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
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=min(lgbm_params.get("max_depth", 6) + 4, 14),
        min_samples_leaf=lgbm_params.get("min_child_samples", 20),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return {"lgbm": lgbm, "xgb": xgb_clf, "rf": rf, "et": et}


def _compute_sample_weights(
    n_samples: int,
    half_life_days: int = 7,
    rows_per_hour: int = 1,
) -> np.ndarray:
    """Pesos exponenciales decrecientes: datos recientes pesan mas.

    El peso se duplica cada *half_life_days* dias.  *rows_per_hour*
    indica cuantas filas hay por hora (= n_coins cuando los datos de
    multiples monedas se intercalan por timestamp).
    """
    half_life_rows = half_life_days * 24 * rows_per_hour
    decay = np.log(2) / max(half_life_rows, 1)
    positions = np.arange(n_samples, dtype=float)
    weights = np.exp(decay * (positions - n_samples + 1))
    weights /= weights.mean()
    return weights


def _purged_ts_split(
    n_samples: int,
    n_splits: int = 3,
    gap: int = 0,
    timestamps: pd.DatetimeIndex | None = None,
    gap_hours: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit con gap (purge) entre train y validation.

    Si se proporcionan *timestamps* y *gap_hours* > 0, el purge se
    calcula por tiempo real (eliminando filas de entrenamiento cuyo
    timestamp este a menos de *gap_hours* horas del inicio de
    validacion).  Esto es correcto cuando los datos intercalan
    multiples monedas en el mismo instante.

    Si no hay timestamps, usa *gap* filas como fallback.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    purged_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):
        if timestamps is not None and gap_hours > 0:
            val_start_time = timestamps[val_idx[0]]
            purge_cutoff = val_start_time - pd.Timedelta(hours=gap_hours)
            mask = timestamps[train_idx] <= purge_cutoff
            train_idx = train_idx[mask]
        elif gap > 0:
            train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx
        if len(train_idx) > 0:
            purged_splits.append((train_idx, val_idx))
    return purged_splits


# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

class PricePredictor:
    """Stacking ensemble (LightGBM + XGBoost + RF + ExtraTrees)
    con meta-learner LogisticRegression, optimizacion Optuna
    y feature selection automatica."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._base_models: dict[str, Any] | None = None
        self._meta_learner: LogisticRegression | None = None
        self._feature_cols = get_feature_columns(include_btc=True)
        self._selected_features: list[str] | None = None

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, klines_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el stacking ensemble con datos de multiples monedas.

        Pipeline completo:
        1. Preparar datos (features + labels realistas TP/SL).
        2. Optuna -- buscar mejores hiperparametros.
        3. Feature selection -- eliminar features ruidosas.
        4. Stacking: entrenar base models + meta-learner.

        Extrae BTC del dict para usarlo como features de mercado.
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
        target_pct = self._config.target_pct_change
        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series] = []

        for symbol, df in klines_by_symbol.items():
            featured = compute_features(
                df, btc_df=btc_df,
                market_df=market_df if not market_df.empty else None,
            )
            # Labels realistas: TP/SL simulado en ventana de *horizon* velas
            labels = create_labels(
                featured, target_pct,
                horizon=horizon,
                stop_loss_pct=self._config.stop_loss_pct,
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
        X = X.iloc[sort_order]
        y = y.iloc[sort_order]

        # Guardar timestamps ANTES de reset_index para purge basado en tiempo
        timestamps = X.index.copy()
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        logger.info(
            "Datos: %d muestras | positivas=%.1f%%",
            len(X),
            100 * y.mean(),
        )

        # Ajustar scale_pos_weight de XGBoost al ratio real
        neg_count = int((y == 0).sum())
        pos_count = int((y == 1).sum())
        spw = neg_count / max(pos_count, 1)

        # Sample weights: datos recientes pesan mas.
        # rows_per_hour = n_coins porque los datos estan intercalados por timestamp.
        n_coins = len(klines_by_symbol)
        sample_w = _compute_sample_weights(len(X), rows_per_hour=n_coins)

        # ----------------------------------------------------------
        # Paso 1: Optuna -- buscar mejores hiperparametros
        # ----------------------------------------------------------
        logger.info(
            "Paso 1/4: Optimizando hiperparametros con Optuna (%d trials)...",
            OPTUNA_N_TRIALS,
        )
        best_params = _run_optuna(X, y, timestamps=timestamps)

        # ----------------------------------------------------------
        # Paso 2: Feature selection
        # ----------------------------------------------------------
        # Usar solo datos del primer ~70% (train del ultimo fold) para
        # seleccionar features, evitando data leakage.
        logger.info("Paso 2/4: Seleccionando features...")
        fs_splits = _purged_ts_split(
            len(X), n_splits=3,
            timestamps=timestamps, gap_hours=PURGE_GAP_HOURS,
        )
        fs_train_idx = fs_splits[-1][0] if fs_splits else None
        self._selected_features = _select_features(
            X, y, self._feature_cols, best_params,
            train_idx=fs_train_idx,
        )
        X_sel = X[self._selected_features]

        # ----------------------------------------------------------
        # Paso 3: Evaluar con Purged Time Series Split + Stacking
        # ----------------------------------------------------------
        logger.info("Paso 3/4: Evaluando con Purged TimeSeriesSplit...")

        # Purge basado en timestamps reales: elimina filas de train cuyo
        # timestamp este a menos de 48h del inicio de validacion.
        # Esto es correcto con multiples monedas intercaladas.
        splits = _purged_ts_split(
            len(X_sel), n_splits=3,
            timestamps=timestamps, gap_hours=PURGE_GAP_HOURS,
        )

        auc_scores: list[float] = []
        per_model_auc: dict[str, list[float]] = {
            "lgbm": [], "xgb": [], "rf": [], "et": [],
        }

        # Matrices para OOF probabilities (stacking)
        model_names = ["lgbm", "xgb", "rf", "et"]
        oof_probs = np.full((len(y), len(model_names)), np.nan)

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_train = sample_w[train_idx]

            base_models = _build_base_models(best_params, spw)

            # Entrenar cada modelo base por separado
            fold_probs = np.zeros((len(val_idx), len(model_names)))
            for m_idx, (name, model) in enumerate(base_models.items()):
                model.fit(X_train, y_train, sample_weight=w_train)

                y_prob_i = model.predict_proba(X_val)[:, 1]
                fold_probs[:, m_idx] = y_prob_i
                auc_i = roc_auc_score(y_val, y_prob_i)
                per_model_auc[name].append(auc_i)
                logger.info("  Fold %d -- %s AUC: %.4f", fold, name, auc_i)

                # Guardar OOF probabilities para stacking
                oof_probs[val_idx, m_idx] = y_prob_i

            # AUC del stacking (promedio simple como proxy)
            avg_prob = fold_probs.mean(axis=1)
            auc = roc_auc_score(y_val, avg_prob)
            auc_scores.append(auc)
            logger.info("Fold %d -- Stacking AUC (avg proxy): %.4f", fold, auc)

        # Log de AUC medio por modelo
        for name, scores in per_model_auc.items():
            logger.info("%s AUC medio: %.4f", name, np.mean(scores))

        # Metricas del ultimo fold (las mas honestas)
        last_val_idx = splits[-1][1]
        y_val_last = y.iloc[last_val_idx]

        # Usar promedio de OOF del ultimo fold como prediccion
        last_avg_prob = oof_probs[last_val_idx].mean(axis=1)
        y_pred_cv = (last_avg_prob >= 0.5).astype(int)
        cv_report = classification_report(y_val_last, y_pred_cv, output_dict=True)
        logger.info(
            "Metricas HONESTAS (ultimo fold CV, datos no vistos):\n%s",
            classification_report(y_val_last, y_pred_cv),
        )

        # ----------------------------------------------------------
        # Paso 4: Entrenar modelos finales + meta-learner
        # ----------------------------------------------------------
        logger.info("Paso 4/4: Entrenamiento final (stacking)...")

        # Entrenar modelos base en todos los datos
        self._base_models = _build_base_models(best_params, spw)
        for name, model in self._base_models.items():
            model.fit(X_sel, y, sample_weight=sample_w)

        # Meta-learner: LogisticRegression sobre OOF probabilities
        valid_oof_mask = ~np.isnan(oof_probs).any(axis=1)
        oof_valid = oof_probs[valid_oof_mask]
        y_valid = y.values[valid_oof_mask]

        self._meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        self._meta_learner.fit(oof_valid, y_valid)

        # Log de calibracion del meta-learner
        test_inputs = np.array([
            [0.10, 0.10, 0.10, 0.10],
            [0.15, 0.15, 0.15, 0.15],
            [0.20, 0.20, 0.20, 0.20],
            [0.25, 0.25, 0.25, 0.25],
            [0.30, 0.30, 0.30, 0.30],
            [0.40, 0.40, 0.40, 0.40],
            [0.50, 0.50, 0.50, 0.50],
        ])
        test_outputs = self._meta_learner.predict_proba(test_inputs)[:, 1]
        logger.info("Meta-learner calibracion (4 modelos de acuerdo):")
        for inp, out in zip(test_inputs, test_outputs):
            logger.info(
                "  base_probs=%.2f -> meta=%.1f%%", inp[0], out * 100,
            )

        mean_auc = float(np.mean(auc_scores))
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
        for name, scores in per_model_auc.items():
            metrics[f"auc_{name}"] = float(np.mean(scores))

        logger.info("Stacking AUC medio CV: %.4f", mean_auc)
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

        Usa los 4 modelos base para generar probabilidades y el
        meta-learner para combinarlas en una probabilidad calibrada.
        """
        if self._base_models is None:
            raise RuntimeError("El modelo no esta entrenado ni cargado.")

        btc_df = klines_by_symbol.get(f"BTC{_QUOTE}")

        # Calcular features de mercado global para prediccion
        market_df = compute_market_features(klines_by_symbol)
        mkt = market_df if not market_df.empty else None

        feat_cols = self._selected_features or self._feature_cols
        model_names = list(self._base_models.keys())

        predictions: dict[str, float] = {}
        for symbol, df in klines_by_symbol.items():
            try:
                featured = compute_features(df, btc_df=btc_df, market_df=mkt)
                if featured.empty:
                    continue
                last_row = featured[feat_cols].iloc[[-1]]

                # Obtener probabilidades de cada modelo base
                base_probs = np.array([
                    self._base_models[name].predict_proba(last_row)[0, 1]
                    for name in model_names
                ]).reshape(1, -1)

                # Meta-learner combina las probabilidades
                if self._meta_learner is not None:
                    prob = float(
                        self._meta_learner.predict_proba(base_probs)[0, 1],
                    )
                else:
                    prob = float(base_probs.mean())

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
        """Guarda los modelos base, meta-learner y features seleccionadas."""
        save_path = path or MODEL_FILE
        if self._base_models is None:
            raise RuntimeError("No hay modelo para guardar.")
        payload = {
            "base_models": self._base_models,
            "meta_learner": self._meta_learner,
            "selected_features": self._selected_features,
        }
        joblib.dump(payload, save_path)
        logger.info("Stacking ensemble guardado en %s", save_path)
        return save_path

    def load(self, path: Path | None = None) -> None:
        """Carga un ensemble guardado desde disco."""
        load_path = path or MODEL_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"No se encontro el modelo en {load_path}")
        payload = joblib.load(load_path)
        if isinstance(payload, dict) and "base_models" in payload:
            self._base_models = payload["base_models"]
            self._meta_learner = payload.get("meta_learner")
            self._selected_features = payload.get("selected_features")
        elif isinstance(payload, dict) and "model" in payload:
            # Compatibilidad con modelos VotingClassifier antiguos
            old_model = payload["model"]
            self._base_models = {
                name: est
                for name, est in old_model.named_estimators_.items()
            }
            self._meta_learner = payload.get("calibrator")
            self._selected_features = payload.get("selected_features")
        else:
            raise ValueError("Formato de modelo no reconocido.")
        logger.info("Ensemble cargado desde %s", load_path)

    @property
    def is_trained(self) -> bool:
        return self._base_models is not None

    def feature_importance(self) -> pd.DataFrame:
        """Devuelve importancia media de cada feature (promediada entre modelos).

        LightGBM, XGBoost, RF y ExtraTrees exponen feature_importances_.
        Promediamos las cuatro (normalizadas).
        """
        if self._base_models is None:
            raise RuntimeError("Modelo no entrenado.")

        feat_cols = self._selected_features or self._feature_cols
        importances: list[np.ndarray] = []
        for _name, estimator in self._base_models.items():
            imp = np.array(estimator.feature_importances_, dtype=float)
            total = imp.sum()
            if total > 0:
                imp = imp / total
                importances.append(imp)

        if not importances:
            return pd.DataFrame({"feature": feat_cols, "importance": 0.0})
        avg_importance = np.mean(importances, axis=0)
        return (
            pd.DataFrame({"feature": feat_cols, "importance": avg_importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
