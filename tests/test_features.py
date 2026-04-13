"""Tests para el modulo de calculo de features tecnicas."""

import numpy as np
import pandas as pd

from src.data.features import compute_features, get_feature_columns


def _make_klines(n: int = 200, seed: int = 42, base_price: float = 100.0) -> pd.DataFrame:
    """Genera un DataFrame de klines sintetico para testing."""
    rng = np.random.default_rng(seed)
    close = base_price + np.cumsum(rng.normal(0, 1, n))
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


def _make_btc_klines(n: int = 200) -> pd.DataFrame:
    """Genera un DataFrame de BTC klines sintetico para testing."""
    return _make_klines(n=n, seed=99, base_price=60000.0)


class TestComputeFeatures:
    def test_returns_dataframe(self) -> None:
        df = _make_klines()
        btc = _make_btc_klines()
        result = compute_features(df, btc_df=btc)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_values(self) -> None:
        df = _make_klines()
        btc = _make_btc_klines()
        result = compute_features(df, btc_df=btc)
        assert not result.isna().any().any()

    def test_no_inf_values(self) -> None:
        df = _make_klines()
        btc = _make_btc_klines()
        result = compute_features(df, btc_df=btc)
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()

    def test_has_expected_columns_with_btc(self) -> None:
        df = _make_klines()
        btc = _make_btc_klines()
        result = compute_features(df, btc_df=btc)
        expected = get_feature_columns(include_btc=True)
        for col in expected:
            assert col in result.columns, f"Falta columna: {col}"

    def test_has_expected_columns_without_btc(self) -> None:
        df = _make_klines()
        result = compute_features(df)
        expected = get_feature_columns(include_btc=False)
        for col in expected:
            assert col in result.columns, f"Falta columna: {col}"

    def test_preserves_original_columns(self) -> None:
        df = _make_klines()
        btc = _make_btc_klines()
        result = compute_features(df, btc_df=btc)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_fewer_rows_after_dropna(self) -> None:
        df = _make_klines()
        btc = _make_btc_klines()
        result = compute_features(df, btc_df=btc)
        assert len(result) < len(df)

    def test_feature_column_list_matches(self) -> None:
        df = _make_klines()
        btc = _make_btc_klines()
        result = compute_features(df, btc_df=btc)
        cols = get_feature_columns(include_btc=True)
        for c in cols:
            assert c in result.columns

    def test_works_without_btc_data(self) -> None:
        df = _make_klines()
        result = compute_features(df, btc_df=None)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # BTC columns should NOT be present
        btc_cols = [
            "btc_roc_1", "btc_roc_6", "btc_roc_12", "btc_roc_24",
            "btc_rsi_14", "btc_volatility_24", "coin_btc_corr_24",
        ]
        for col in btc_cols:
            assert col not in result.columns

    def test_empty_for_insufficient_rows(self) -> None:
        df = _make_klines(n=30)
        result = compute_features(df)
        assert result.empty


class TestGetFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        cols = get_feature_columns(include_btc=True)
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)

    def test_no_duplicates(self) -> None:
        cols = get_feature_columns(include_btc=True)
        assert len(cols) == len(set(cols))

    def test_btc_columns_included(self) -> None:
        cols_with = get_feature_columns(include_btc=True)
        cols_without = get_feature_columns(include_btc=False)
        assert len(cols_with) > len(cols_without)
        assert "btc_roc_1" in cols_with
        assert "btc_roc_1" not in cols_without
