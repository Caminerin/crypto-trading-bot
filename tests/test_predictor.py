"""Tests para el modulo de prediccion (model/predictor.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.predictor import (
    _compute_sample_weights,
    _purged_ts_split,
    create_labels,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 42, base: float = 100.0) -> pd.DataFrame:
    """Genera un DataFrame OHLCV sintetico con indice temporal."""
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(1000, 10000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


# ------------------------------------------------------------------
# create_labels
# ------------------------------------------------------------------

class TestCreateLabels:
    def test_returns_series_same_length(self) -> None:
        df = _make_ohlcv()
        labels = create_labels(df, target_pct=0.05, horizon=48, stop_loss_pct=0.05)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(df)

    def test_last_horizon_rows_are_nan(self) -> None:
        df = _make_ohlcv(n=100)
        labels = create_labels(df, target_pct=0.05, horizon=48, stop_loss_pct=0.05)
        # Las ultimas `horizon` filas no tienen suficientes velas futuras
        assert labels.iloc[-1] is np.nan or pd.isna(labels.iloc[-1])

    def test_labels_are_binary(self) -> None:
        df = _make_ohlcv(n=200)
        labels = create_labels(df, target_pct=0.03, horizon=24, stop_loss_pct=0.03)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_guaranteed_tp_hit(self) -> None:
        """Si el precio sube mucho en la siguiente vela, la label debe ser 1."""
        n = 60
        df = _make_ohlcv(n=n, base=100.0)
        # Forzar que la vela 1 tenga un high muy alto respecto al close de vela 0
        df.iloc[0, df.columns.get_loc("close")] = 100.0
        df.iloc[1, df.columns.get_loc("high")] = 200.0  # +100%
        df.iloc[1, df.columns.get_loc("low")] = 99.0    # no toca SL
        labels = create_labels(df, target_pct=0.05, horizon=48, stop_loss_pct=0.05)
        assert labels.iloc[0] == 1.0

    def test_guaranteed_sl_hit(self) -> None:
        """Si el precio cae mucho antes de subir, la label debe ser 0."""
        n = 60
        df = _make_ohlcv(n=n, base=100.0)
        df.iloc[0, df.columns.get_loc("close")] = 100.0
        df.iloc[1, df.columns.get_loc("low")] = 50.0     # -50% toca SL
        df.iloc[1, df.columns.get_loc("high")] = 100.5    # no toca TP
        labels = create_labels(df, target_pct=0.05, horizon=48, stop_loss_pct=0.05)
        assert labels.iloc[0] == 0.0

    def test_no_stop_loss_only_checks_tp(self) -> None:
        df = _make_ohlcv(n=60, base=100.0)
        df.iloc[0, df.columns.get_loc("close")] = 100.0
        df.iloc[1, df.columns.get_loc("high")] = 200.0
        df.iloc[1, df.columns.get_loc("low")] = 50.0  # sin SL, no importa
        labels = create_labels(df, target_pct=0.05, horizon=48, stop_loss_pct=None)
        assert labels.iloc[0] == 1.0


# ------------------------------------------------------------------
# _purged_ts_split
# ------------------------------------------------------------------

class TestPurgedTsSplit:
    def test_returns_correct_number_of_splits(self) -> None:
        splits = _purged_ts_split(1000, n_splits=3)
        assert len(splits) == 3

    def test_val_after_train(self) -> None:
        """Validacion debe ser siempre posterior a train (cronologico)."""
        splits = _purged_ts_split(500, n_splits=3)
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()

    def test_row_gap_removes_rows(self) -> None:
        splits_no_gap = _purged_ts_split(500, n_splits=3, gap=0)
        splits_gap = _purged_ts_split(500, n_splits=3, gap=48)
        # Con gap, los train deben tener menos filas
        for (t_no, _), (t_gap, _) in zip(splits_no_gap, splits_gap):
            assert len(t_gap) < len(t_no)

    def test_timestamp_based_purge(self) -> None:
        """Con timestamps y gap_hours, el purge debe basarse en tiempo real."""
        n = 600
        # Simular 3 monedas intercaladas: misma hora, 3 filas
        base_times = pd.date_range("2024-01-01", periods=n // 3, freq="h", tz="UTC")
        timestamps = pd.DatetimeIndex(
            [t for t in base_times for _ in range(3)]
        )
        assert len(timestamps) == n

        splits = _purged_ts_split(
            n, n_splits=3,
            timestamps=timestamps, gap_hours=48,
        )
        for train_idx, val_idx in splits:
            train_max_time = timestamps[train_idx[-1]]
            val_min_time = timestamps[val_idx[0]]
            gap_hours = (val_min_time - train_max_time).total_seconds() / 3600
            # El gap debe ser >= 48 horas (puede ser ligeramente mas)
            assert gap_hours >= 48, (
                f"Gap insuficiente: {gap_hours:.1f}h < 48h"
            )

    def test_timestamp_purge_more_aggressive_than_row(self) -> None:
        """Con monedas intercaladas, el purge por tiempo elimina mas filas."""
        n = 600
        base_times = pd.date_range("2024-01-01", periods=n // 3, freq="h", tz="UTC")
        timestamps = pd.DatetimeIndex(
            [t for t in base_times for _ in range(3)]
        )
        splits_row = _purged_ts_split(n, n_splits=3, gap=48)
        splits_ts = _purged_ts_split(
            n, n_splits=3,
            timestamps=timestamps, gap_hours=48,
        )
        # Con 3 monedas, 48 filas = 16h, pero 48h de gap por timestamp
        # elimina ~144 filas en vez de 48
        for (t_row, _), (t_ts, _) in zip(splits_row, splits_ts):
            assert len(t_ts) < len(t_row), (
                "Purge por timestamp deberia eliminar mas filas que por fila"
            )


# ------------------------------------------------------------------
# _compute_sample_weights
# ------------------------------------------------------------------

class TestComputeSampleWeights:
    def test_returns_correct_length(self) -> None:
        w = _compute_sample_weights(100)
        assert len(w) == 100

    def test_last_weight_highest(self) -> None:
        """Los datos mas recientes (ultimos) deben tener mayor peso."""
        w = _compute_sample_weights(200)
        assert w[-1] > w[0]

    def test_mean_is_one(self) -> None:
        """Los pesos estan normalizados a media 1."""
        w = _compute_sample_weights(500)
        assert abs(w.mean() - 1.0) < 1e-6

    def test_rows_per_hour_affects_decay(self) -> None:
        """Con mas filas por hora, el decaimiento es mas lento."""
        w1 = _compute_sample_weights(300, rows_per_hour=1)
        w75 = _compute_sample_weights(300, rows_per_hour=75)
        # Con 75 monedas el ratio ultimo/primero debe ser menor
        ratio_1 = w1[-1] / w1[0]
        ratio_75 = w75[-1] / w75[0]
        assert ratio_75 < ratio_1
