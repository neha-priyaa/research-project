"""
TIP-PCA Aligned Visualization Module.

Generates chart outputs for:
- Volatility matrix heatmap (D x n)
- U-shaped intraday pattern
- HAR covariates visualization
- Low-rank structure analysis
- SVD decomposition

Based on:
Choi & Kim (2024) - "Matrix-based Prediction Approach for Intraday
Instantaneous Volatility Vector" arXiv:2403.02591
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple


class TIPCAVisualizer:
    """
    Visualizer for TIP-PCA aligned volatility data.
    """

    def __init__(self, output_dir: str, figsize: tuple = (12, 8)):
        self.output_dir = output_dir
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)

    def plot_volatility_matrix_heatmap(
        self,
        vol_matrix: np.ndarray,
        title: str = "Instantaneous Volatility Matrix (D x n)",
        filename: str = "volatility_matrix_heatmap.png",
    ) -> plt.Figure:
        """Plot heatmap of D x n volatility matrix."""
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(vol_matrix, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xlabel("Intraday Time Point (n)")
        ax.set_ylabel("Day (D)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Instantaneous Volatility")

        n_days, n_intraday = vol_matrix.shape
        ax.set_xlim(0, n_intraday)
        ax.set_ylim(0, n_days)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {filename}")
        plt.close()
        return fig

    def plot_u_shaped_pattern(
        self,
        vol_matrix: np.ndarray,
        title: str = "Average Intraday Volatility Pattern (U-shaped)",
        filename: str = "u_shaped_pattern.png",
    ) -> plt.Figure:
        """Plot average intraday volatility showing U-shaped pattern."""
        vol_matrix = np.nan_to_num(vol_matrix, nan=1e-8, posinf=1e-8, neginf=-1e-8)

        fig, ax = plt.subplots(figsize=self.figsize)

        avg_intraday = np.mean(vol_matrix, axis=0)
        std_intraday = np.std(vol_matrix, axis=0)

        t = np.linspace(0, 1, len(avg_intraday))
        ax.plot(t, avg_intraday, "b-", linewidth=2, label="Mean")
        ax.fill_between(
            t,
            avg_intraday - std_intraday,
            avg_intraday + std_intraday,
            alpha=0.3,
            color="blue",
            label="±1 Std",
        )

        ax.set_xlabel("Normalized Intraday Time")
        ax.set_ylabel("Average Volatility")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {filename}")
        plt.close()
        return fig

    def plot_har_covariates(
        self,
        har_cov: np.ndarray,
        title: str = "HAR Covariates Over Time",
        filename: str = "har_covariates.png",
    ) -> plt.Figure:
        """Plot HAR (daily, weekly, monthly) realized volatility."""
        har_cov = np.nan_to_num(har_cov, nan=0.0, posinf=0.0, neginf=0.0)

        fig, ax = plt.subplots(figsize=self.figsize)

        days = np.arange(len(har_cov))
        ax.plot(days, har_cov[:, 0], "b-", linewidth=1.5, label="RV_daily", alpha=0.8)
        ax.plot(days, har_cov[:, 1], "g-", linewidth=2, label="RV_weekly", alpha=0.8)
        ax.plot(days, har_cov[:, 2], "r-", linewidth=2.5, label="RV_monthly", alpha=0.8)

        ax.set_xlabel("Day")
        ax.set_ylabel("Realized Volatility")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {filename}")
        plt.close()
        return fig

    def plot_svd_analysis(
        self,
        vol_matrix: np.ndarray,
        title: str = "SVD Analysis - Singular Values",
        filename: str = "svd_analysis.png",
    ) -> plt.Figure:
        """Plot singular value decomposition showing low-rank structure."""
        vol_matrix = np.nan_to_num(vol_matrix, nan=1e-8, posinf=1e-8, neginf=-1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        try:
            U, s, Vt = np.linalg.svd(vol_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"  Warning: SVD failed for {filename}, skipping...")
            plt.close()
            return None

        ax1 = axes[0]
        ax1.bar(range(1, len(s) + 1), s, color="steelblue", alpha=0.7)
        ax1.set_xlabel("Singular Value Index")
        ax1.set_ylabel("Singular Value")
        ax1.set_title("Singular Values")
        ax1.grid(True, alpha=0.3)

        cumvar = np.cumsum(s**2) / np.sum(s**2)
        ax2 = axes[1]
        ax2.plot(range(1, len(cumvar) + 1), cumvar, "g-", linewidth=2)
        ax2.axhline(0.95, color="red", linestyle="--", label="95% variance")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Variance Explained")
        ax2.set_title("Cumulative Variance")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {filename}")
        plt.close()
        return fig

    def plot_low_rank_reconstruction(
        self,
        vol_matrix: np.ndarray,
        ranks: list = [1, 3, 5, 10],
        title: str = "Low-Rank Reconstruction Comparison",
        filename: str = "low_rank_reconstruction.png",
    ) -> plt.Figure:
        """Show how different ranks reconstruct the original matrix."""
        vol_matrix = np.nan_to_num(vol_matrix, nan=1e-8, posinf=1e-8, neginf=-1e-8)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        try:
            U, s, Vt = np.linalg.svd(vol_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"  Warning: SVD failed for {filename}, skipping...")
            plt.close()
            return None

        idx = 0
        for r in ranks:
            ax = axes[idx // 2, idx % 2]

            s_r = np.diag(s[:r])
            reconstructed = U[:, :r] @ s_r @ Vt[:r, :]

            im = ax.imshow(reconstructed, aspect="auto", cmap="YlOrRd", origin="lower")
            ax.set_title(f"Rank {r} Reconstruction")
            ax.set_xlabel("Intraday Point")
            ax.set_ylabel("Day")
            plt.colorbar(im, ax=ax, label="Volatility")

            error = np.linalg.norm(vol_matrix - reconstructed) / np.linalg.norm(
                vol_matrix
            )
            ax.text(
                0.02,
                0.98,
                f"Rel. Error: {error:.2%}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white"),
            )

            idx += 1

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {filename}")
        plt.close()
        return fig

    def plot_interday_dynamics(
        self,
        vol_matrix: np.ndarray,
        har_cov: Optional[np.ndarray] = None,
        title: str = "Interday Volatility Dynamics",
        filename: str = "interday_dynamics.png",
    ) -> plt.Figure:
        """Plot interday volatility evolution showing autoregressive structure."""
        vol_matrix = np.nan_to_num(vol_matrix, nan=1e-8, posinf=1e-8, neginf=-1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        daily_rv = np.sqrt(np.sum(vol_matrix**2, axis=1))

        ax1 = axes[0]
        ax1.plot(daily_rv, "b-", linewidth=1.5, alpha=0.8)
        ax1.axhline(
            np.mean(daily_rv),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(daily_rv):.4f}",
        )
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Daily Realized Volatility")
        ax1.set_title("Daily RV Time Series")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.acorr(daily_rv, maxlags=30, usevlines=True, normed=True)
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Autocorrelation")
        ax2.set_title("Autocorrelation of Daily RV")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {filename}")
        plt.close()
        return fig

    def plot_volatility_right_singular_vectors(
        self,
        vol_matrix: np.ndarray,
        title: str = "Right Singular Vectors (Intraday Patterns)",
        filename: str = "right_singular_vectors.png",
    ) -> plt.Figure:
        """Plot right singular vectors showing intraday patterns."""
        vol_matrix = np.nan_to_num(vol_matrix, nan=1e-8, posinf=1e-8, neginf=-1e-8)

        fig, ax = plt.subplots(figsize=self.figsize)

        try:
            U, s, Vt = np.linalg.svd(vol_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"  Warning: SVD failed for {filename}, skipping...")
            plt.close()
            return None

        t = np.linspace(0, 1, vol_matrix.shape[1])
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(Vt))))

        for i in range(min(5, len(Vt))):
            ax.plot(
                t,
                Vt[i],
                color=colors[i],
                linewidth=1.5,
                label=f"SV {i+1} (σ={s[i]:.4f})",
                alpha=0.8,
            )

        ax.set_xlabel("Normalized Intraday Time")
        ax.set_ylabel("Singular Vector Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {filename}")
        plt.close()
        return fig

    def plot_summary_dashboard(
        self,
        vol_matrix: np.ndarray,
        har_cov: Optional[np.ndarray] = None,
        prefix: str = "tipca_analysis",
    ) -> None:
        """Generate complete dashboard of all visualizations."""
        print("\nGenerating TIP-PCA visualizations...")

        self.plot_volatility_matrix_heatmap(
            vol_matrix, filename=f"{prefix}_1_matrix_heatmap.png"
        )

        self.plot_u_shaped_pattern(
            vol_matrix, filename=f"{prefix}_2_u_shaped_pattern.png"
        )

        if har_cov is not None:
            self.plot_har_covariates(har_cov, filename=f"{prefix}_3_har_covariates.png")

        self.plot_svd_analysis(vol_matrix, filename=f"{prefix}_4_svd_analysis.png")

        self.plot_low_rank_reconstruction(
            vol_matrix, filename=f"{prefix}_5_low_rank.png"
        )

        self.plot_interday_dynamics(
            vol_matrix, har_cov, filename=f"{prefix}_6_interday_dynamics.png"
        )

        self.plot_volatility_right_singular_vectors(
            vol_matrix, filename=f"{prefix}_7_right_sv.png"
        )

        print(f"\nAll visualizations saved to: {self.output_dir}")


def visualize_simulated_data(
    n_days: int = 100, n_intraday: int = 39, output_dir: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate visualizations for simulated TIP-PCA data.
    """
    from simulator_tipca import TIPCASimulator

    if output_dir is None:
        output_dir = "/Users/nehapriya/Desktop/research_project/data/processed"

    simulator = TIPCASimulator(n_days=n_days, n_intraday=n_intraday, rank=3, seed=42)
    vol_matrix = simulator.get_volatility_matrix()
    har_cov = simulator.generate_har_covariates()

    visualizer = TIPCAVisualizer(output_dir)
    visualizer.plot_summary_dashboard(vol_matrix, har_cov, prefix="simulated")

    return vol_matrix, har_cov


def visualize_nifty_data(
    nifty_csv_path: str, output_dir: str = None, n_intraday: int = 39
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate visualizations for real NIFTY data transformed with TIP-PCA.
    """
    from transform_nifty_tipca import TIPCATransformer

    if output_dir is None:
        output_dir = "/Users/nehapriya/Desktop/research_project/data/processed"

    df = pd.read_csv(nifty_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    transformer = TIPCATransformer(n_intraday=n_intraday)
    flat_df, vol_matrix = transformer.transform_to_flat_dataframe(
        df, "close", "timestamp"
    )

    har_cov = transformer.compute_har_covariates(
        flat_df.groupby("day")["realized_vol"].first().values
    )

    visualizer = TIPCAVisualizer(output_dir)
    visualizer.plot_summary_dashboard(vol_matrix, har_cov, prefix="nifty")

    return vol_matrix, flat_df


if __name__ == "__main__":
    output_dir = "/Users/nehapriya/Desktop/research_project/data/processed"

    print("=" * 60)
    print("TIP-PCA Visualization Generator")
    print("=" * 60)

    vol_matrix, har_cov = visualize_simulated_data(n_days=100, n_intraday=39)

    nifty_path = "/Users/nehapriya/Downloads/nifty_underlying_minute.csv"
    if os.path.exists(nifty_path):
        print("\nGenerating NIFTY visualizations...")
        vol_nifty, df_nifty = visualize_nifty_data(nifty_path)
    else:
        print(f"\nNIFTY data not found at {nifty_path}")
        print("Skipping NIFTY visualization.")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
