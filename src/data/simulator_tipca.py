"""
TIP-PCA Aligned Data Simulator for Intraday Volatility Prediction.

Implements the instantaneous volatility matrix generation based on:
Choi & Kim (2024) - "Matrix-based Prediction Approach for Intraday
Instantaneous Volatility Vector" arXiv:2403.02591
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TIPCASimulator:
    """
    Simulates instantaneous volatility matrix with low-rank structure
    matching TIP-PCA methodology.
    """

    def __init__(
        self,
        n_days: int = 252,
        n_intraday: int = 39,
        rank: int = 3,
        noise_level: float = 0.1,
        seed: Optional[int] = 42,
    ):
        self.n_days = n_days
        self.n_intraday = n_intraday
        self.rank = rank
        self.noise_level = noise_level
        self.seed = seed

    def generate_har_covariates(self) -> np.ndarray:
        """
        Generate HAR (Heterogeneous Autoregressive) covariates.
        Per the paper: RV_{i-1}, RV_{i-5}, RV_{i-22} (daily, weekly, monthly).
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        daily_rv = np.random.exponential(scale=0.01, size=self.n_days)

        weekly_rv = np.zeros(self.n_days)
        for i in range(5, self.n_days):
            weekly_rv[i] = np.mean(daily_rv[i - 5:i])

        monthly_rv = np.zeros(self.n_days)
        for i in range(22, self.n_days):
            monthly_rv[i] = np.mean(daily_rv[i - 22:i])

        return np.column_stack([daily_rv, weekly_rv, monthly_rv])

    def generate_u_shaped_basis(self) -> np.ndarray:
        """
        Generate U-shaped intraday basis functions.
        Volatility follows pattern: higher at open/close, lower in middle.
        Paper models this as: a1*(t-a2)^2 + a3
        """
        t = np.linspace(0, 1, self.n_intraday)

        u_shaped = np.zeros((self.n_intraday, 3))
        u_shaped[:, 0] = t**2
        u_shaped[:, 1] = t
        u_shaped[:, 2] = np.ones(self.n_intraday)

        return u_shaped

    def generate_low_rank_volatility_matrix(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate low-rank instantaneous volatility matrix following TIP-PCA:
        Sigma = U * Lambda * V^T + Noise

        Returns:
            U: Left singular vectors (n_days x rank) - interday dynamics
            V: Right singular vectors (n_intraday x rank) - intraday patterns
            Lambda: Singular values (rank x rank)
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        har_cov = self.generate_har_covariates()

        u_basis = np.random.randn(self.n_days, self.rank)
        for k in range(self.rank):
            u_basis[:, k] = (
                0.6 * har_cov[:, 0] * (1 + 0.1 * k)
                + 0.3 * har_cov[:, 1] * np.random.randn()
                + 0.1 * har_cov[:, 2]
            )
            u_basis[:, k] = u_basis[:, k] - np.mean(u_basis[:, k])
            u_basis[:, k] = u_basis[:, k] / (np.std(u_basis[:, k]) + 1e-8)

        u, _ = np.linalg.qr(u_basis)

        t = np.linspace(0, 1, self.n_intraday)
        v_basis = np.zeros((self.n_intraday, self.rank))
        v_basis[:, 0] = 1.0 - 1.5 * (t - 0.5) ** 2
        v_basis[:, 1] = np.sin(2 * np.pi * t)
        v_basis[:, 2] = np.cos(4 * np.pi * t)

        for k in range(self.rank):
            v_basis[:, k] = v_basis[:, k] - np.mean(v_basis[:, k])
            v_basis[:, k] = v_basis[:, k] / (np.std(v_basis[:, k]) + 1e-8)

        v, _ = np.linalg.qr(v_basis)

        lambda_vals = np.array([0.02, 0.015, 0.01])[: self.rank] * (
            1 + 0.2 * np.random.randn(self.rank)
        )
        Lambda = np.diag(np.abs(lambda_vals))

        return u, v, Lambda

    def generate_volatility_matrix(self) -> np.ndarray:
        """
        Generate the full D x n instantaneous volatility matrix.
        D = n_days, n = n_intraday
        """
        u, v, Lambda = self.generate_low_rank_volatility_matrix()

        low_rank = u @ Lambda @ v.T

        noise = (
            np.random.randn(self.n_days, self.n_intraday)
            * self.noise_level
            * np.std(low_rank)
        )

        volatility_matrix = np.abs(low_rank + noise)
        volatility_matrix = np.maximum(volatility_matrix, 1e-8)

        return volatility_matrix

    def generate_realized_volatility(
        self, volatility_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute daily realized volatility (sum of squared instantaneous vol).
        This matches the HAR model covariates.
        """
        daily_rv = np.sum(volatility_matrix**2, axis=1)
        return np.sqrt(daily_rv)

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete dataset in flat format (for compatibility)
        while also storing matrix format internally.
        """
        volatility_matrix = self.generate_volatility_matrix()
        daily_rv = self.generate_realized_volatility(volatility_matrix)

        data = []
        for day in range(self.n_days):
            for intraday in range(self.n_intraday):
                t_normalized = intraday / self.n_intraday
                data.append(
                    {
                        "day": day,
                        "intraday": intraday,
                        "time_normalized": t_normalized,
                        "instantaneous_vol": volatility_matrix[day, intraday],
                        "realized_vol": daily_rv[day],
                    }
                )

        return pd.DataFrame(data), volatility_matrix

    def get_volatility_matrix(self) -> np.ndarray:
        """Return the D x n volatility matrix directly."""
        return self.generate_volatility_matrix()


def simulate_tipca_data(
    n_days: int = 252,
    n_intraday: int = 39,
    rank: int = 3,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience function to generate TIP-PCA aligned data.

    Returns:
        df: Flat DataFrame with day, intraday, time_normalized, vol
        matrix: D x n volatility matrix (n_days x n_intraday)
    """
    simulator = TIPCASimulator(
        n_days=n_days,
        n_intraday=n_intraday,
        rank=rank,
        noise_level=noise_level,
        seed=seed,
    )
    return simulator.generate_dataset()


if __name__ == "__main__":
    df, vol_matrix = simulate_tipca_data(n_days=100, n_intraday=39)

    print(f"Volatility Matrix Shape: {vol_matrix.shape}")
    print(f"DataFrame Shape: {df.shape}")
    print("\nMatrix (first 5 days, first 10 intraday points):")
    print(vol_matrix[:5, :10])
    print(f"\nLow-rank approximation rank: {3}")  # noqa: F541
    print(f"Effective rank: {np.linalg.matrix_rank(vol_matrix)}")
