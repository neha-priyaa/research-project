"""
TIP-PCA Aligned Data Transformer for NIFTY high-frequency data.

Implements:
- Jump-robust pre-averaging volatility estimation (Figueroa-López & Wu, 2024)
- D x n matrix structure for TIP-PCA
- HAR covariate computation
- U-shaped pattern extraction

Based on:
Choi & Kim (2024) - "Matrix-based Prediction Approach for Intraday
Instantaneous Volatility Vector" arXiv:2403.02591
"""

import numpy as np
import pandas as pd
from typing import Tuple


class TIPCATransformer:
    """
    Transform high-frequency data to TIP-PCA compatible format.
    """

    def __init__(
        self,
        n_intraday: int = 39,
        preavg_window: int = 10,
        sampling_freq: str = "1min",
    ):
        self.n_intraday = n_intraday
        self.preavg_window = preavg_window
        self.sampling_freq = sampling_freq

    def preaveraging_estimator(
        self, prices: np.ndarray, theta: float = 0.25
    ) -> np.ndarray:
        """
        Jump-robust pre-averaging estimator for instantaneous volatility.

        Based on Figueroa-López & Wu (2024):
        Uses rolling average to filter out microstructure noise and jumps.

        Args:
            prices: Raw price series
            theta: Smoothing parameter (0 < theta < 0.5)

        Returns:
            Estimated instantaneous volatility
        """
        n = len(prices)
        k = int(n * theta)

        returns = np.diff(np.log(prices))

        preavg = np.zeros(n - k)
        for j in range(n - k):
            preavg[j] = np.mean(returns[j:j + k])

        squared = preavg**2

        vol_estimate = np.zeros(n)
        for i in range(k, n):
            vol_estimate[i] = np.sum(squared[i - k:i]) / k

        vol_estimate = np.sqrt(np.maximum(vol_estimate, 1e-10))
        vol_estimate[:k] = vol_estimate[k]

        return vol_estimate

    def compute_realized_volatility(self, prices: np.ndarray) -> float:
        """
        Compute daily realized volatility (sum of squared returns).
        """
        returns = np.diff(np.log(prices))
        rv = np.sqrt(np.sum(returns**2))
        return rv

    def compute_har_covariates(
        self, daily_rv: np.ndarray, lookback: Tuple[int, int] = (5, 22)
    ) -> np.ndarray:
        """
        Compute HAR covariates: RV_{i-1}, RV_{i-5}, RV_{i-22}.

        Args:
            daily_rv: Array of daily realized volatilities
            lookback: (weekly_lookback, monthly_lookback)

        Returns:
            Array with columns: [RV_daily, RV_weekly, RV_monthly]
        """
        n_days = len(daily_rv)
        weekly_lb, monthly_lb = lookback

        har_cov = np.zeros((n_days, 3))
        har_cov[:, 0] = daily_rv

        for i in range(weekly_lb, n_days):
            har_cov[i, 1] = np.mean(daily_rv[i - weekly_lb:i])

        for i in range(monthly_lb, n_days):
            har_cov[i, 2] = np.mean(daily_rv[i - monthly_lb:i])

        if weekly_lb < n_days:
            har_cov[:weekly_lb, 1] = har_cov[weekly_lb, 1]

        if monthly_lb < n_days:
            har_cov[:monthly_lb, 2] = har_cov[monthly_lb, 2]

        return har_cov

    def fit_u_shaped_pattern(self, intraday_vol: np.ndarray) -> dict:
        """
        Fit U-shaped intraday pattern using quadratic basis.
        Pattern: a1*(t-a2)^2 + a3

        Returns:
            Dictionary with fitted parameters
        """
        n = len(intraday_vol)
        t = np.linspace(0, 1, n)

        X = np.column_stack([t**2, t, np.ones(n)])

        coeffs, residuals, rank, s = np.linalg.lstsq(
            X, intraday_vol, rcond=None
        )

        fitted = X @ coeffs

        return {
            "coeffs": coeffs,
            "a1": coeffs[0],
            "a2": -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else 0.5,
            "a3": coeffs[2],
            "fitted": fitted,
            "r_squared": 1
            - np.sum((intraday_vol - fitted) ** 2) / np.var(intraday_vol),
        }

    def transform_to_matrix(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        time_col: str = "timestamp",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transform high-frequency data to D x n volatility matrix.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close,
                volume
            price_col: Column name for prices
            time_col: Column name for timestamps

        Returns:
            vol_matrix: D x n instantaneous volatility matrix
            summary_df: DataFrame with daily info and HAR covariates
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        df["date"] = pd.to_datetime(df[time_col]).dt.date

        unique_dates = df["date"].unique()
        n_days = len(unique_dates)

        vol_matrix = np.zeros((n_days, self.n_intraday))
        daily_rv = np.zeros(n_days)
        daily_prices = []

        for day_idx, date in enumerate(unique_dates):
            day_data = df[df["date"] == date].copy()
            day_data = day_data.sort_values(time_col).reset_index(drop=True)

            prices = day_data[price_col].values

            if len(prices) >= self.preavg_window:
                vol = self.preaveraging_estimator(prices)
            else:
                returns = np.diff(np.log(prices))
                vol = np.abs(returns)
                vol = np.concatenate([[vol[0]], vol])

            n_points = min(len(vol), self.n_intraday)

            if n_points < self.n_intraday:
                vol_padded = np.zeros(self.n_intraday)
                vol_padded[:n_points] = vol
                vol_padded[n_points:] = vol[-1]
                vol = vol_padded
            else:
                n_pts = self.n_intraday
                idx_range = np.linspace(0, len(vol) - 1, n_pts, dtype=int)
                vol = vol[idx_range]

            vol_matrix[day_idx] = vol
            daily_rv[day_idx] = self.compute_realized_volatility(prices)
            daily_prices.append(prices)

        har_cov = self.compute_har_covariates(daily_rv)

        summary_df = pd.DataFrame(
            {
                "day": range(n_days),
                "date": unique_dates,
                "realized_vol": daily_rv,
                "RV_daily": har_cov[:, 0],
                "RV_weekly": har_cov[:, 1],
                "RV_monthly": har_cov[:, 2],
            }
        )

        return vol_matrix, summary_df

    def transform_to_flat_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Transform to flat DataFrame format (for compatibility).
        """
        vol_matrix, summary_df = self.transform_to_matrix(
            df, price_col, time_col
        )

        flat_data = []
        for day in range(vol_matrix.shape[0]):
            for intraday in range(vol_matrix.shape[1]):
                t_norm = intraday / vol_matrix.shape[1]
                flat_data.append(
                    {
                        "day": day,
                        "intraday": intraday,
                        "time_normalized": t_norm,
                        "instantaneous_vol": vol_matrix[day, intraday],
                    }
                )

        flat_df = pd.DataFrame(flat_data)
        flat_df = flat_df.merge(summary_df, on="day", how="left")

        return flat_df, vol_matrix

    def save_data(
        self,
        vol_matrix: np.ndarray,
        summary_df: pd.DataFrame,
        output_dir: str,
        prefix: str = "nifty",
    ):
        """
        Save volatility matrix and summary data.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        vol_path = f"{output_dir}/{prefix}_volatility_matrix.npy"
        np.save(vol_path, vol_matrix)
        csv_path = f"{output_dir}/{prefix}_daily_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        print(f"Saved: {output_dir}/{prefix}_volatility_matrix.npy")
        print(f"Saved: {output_dir}/{prefix}_daily_summary.csv")
        print(f"Matrix shape: {vol_matrix.shape}")


def transform_nifty_data(
    input_path: str,
    output_dir: str,
    n_intraday: int = 39,
    price_col: str = "close",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience function to transform NIFTY data to TIP-PCA format.
    """
    df = pd.read_csv(input_path)
    df[time_col] = pd.to_datetime(df[time_col])

    transformer = TIPCATransformer(n_intraday=n_intraday)

    flat_df, vol_matrix = transformer.transform_to_flat_dataframe(
        df, price_col, time_col
    )

    transformer.save_data(
        vol_matrix, flat_df.groupby("day").first().reset_index(), output_dir
    )

    return flat_df, vol_matrix


if __name__ == "__main__":
    import os
    from simulator_tipca import TIPCASimulator

    input_path = "/Users/nehapriya/Downloads/nifty_underlying_minute.csv"
    output_dir = "/Users/nehapriya/Desktop/research_project/data/processed"

    if os.path.exists(input_path):
        flat_df, vol_matrix = transform_nifty_data(input_path, output_dir)
        print(f"\nTransformed data shape: {flat_df.shape}")
        print(f"Volatility matrix shape: {vol_matrix.shape}")
        print("\nSample data (first 10 rows):")  # noqa: F541
        print(flat_df.head(10).to_string(index=False))
    else:
        print(f"File not found: {input_path}")
        print("Creating sample data for demonstration...")

        sample_df, sample_matrix = TIPCASimulator(
            n_days=100, n_intraday=39
        ).generate_dataset()
        print(f"\nSample matrix shape: {sample_matrix.shape}")
        print(f"Sample data shape: {sample_df.shape}")
