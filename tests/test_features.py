"""Tests para el módulo de cálculo de features técnicas."""

import numpy as np
import pandas as pd

from src.data.features import compute_features, get_feature_columns


def _make_klines(n: int = 100) -> pd.DataFrame:
    """Genera un DataFrame de klines sintético para testing."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 10000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


class TestComputeFeatures:
    def test_returns_dataframe(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_values(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        assert not result.isna().any().any()

    def test_no_inf_values(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()

    def test_has_expected_columns(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        expected = get_feature_columns()
        for col in expected:
            assert col in result.columns, f"Falta columna: {col}"

    def test_preserves_original_columns(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_fewer_rows_after_dropna(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        # Debe tener menos filas por el cálculo de medias móviles
        assert len(result) < len(df)

    def test_feature_column_list_matches(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        cols = get_feature_columns()
        # Todas las feature columns existen
        for c in cols:
            assert c in result.columns


class TestGetFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)

    def test_no_duplicates(self) -> None:
        cols = get_feature_columns()
        assert len(cols) == len(set(cols))
