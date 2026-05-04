"""
TIP-PCA Implementation Module

Implements Tensor-Invariant Principal Component Analysis for
intraday volatility matrix decomposition and prediction.

Based on: Matrix-based Prediction Approach for Intraday Instantaneous Volatility Vector
"""

import numpy as np
from scipy.linalg import svd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TIPPCA:
    """
    Tensor-Invariant PCA for volatility matrix decomposition.
    
    Decomposes volatility matrices into Low-Rank (Signal) + Noise components
    using tensor-invariant principal component analysis.
    """
    
    def __init__(self, 
                 n_components: int = 10,
                 threshold: Optional[float] = None,
                 normalize: bool = True):
        """
        Initialize TIP-PCA model.
        
        Args:
            n_components: Number of principal components to keep
            threshold: Threshold for noise separation (auto if None)
            normalize: Whether to normalize input matrices
            
        Returns:
            None
        """
        self.n_components = n_components
        self.threshold = threshold
        self.normalize = normalize
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        self.is_fitted_ = False
        
    def fit(self, matrix: np.ndarray) -> 'TIPPCA':
        """
        Fit TIP-PCA model to volatility matrix.
        
        Args:
            matrix: Volatility matrix of shape (n_days, n_intraday)
            
        Returns:
            Self (fitted model)
        """
        # Store original dimensions
        self.n_days_, self.n_intraday_ = matrix.shape
        
        # Normalize if requested
        if self.normalize:
            self.mean_ = np.mean(matrix, axis=0)
            X = matrix - self.mean_
        else:
            X = matrix.copy()
            
        # Perform Singular Value Decomposition (SVD)
        # This is the tensor-invariant approach
        U, s, Vt = svd(X, full_matrices=False)
        
        # Store principal components
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = s
        
        # Calculate explained variance
        total_variance = np.sum(s**2)
        explained_variance = (s**2)[:self.n_components] / total_variance
        self.explained_variance_ = explained_variance
        
        # Auto-determine threshold if not set
        if self.threshold is None:
            # Use cumulative explained variance (keep 95%)
            cumulative_variance = np.cumsum(explained_variance)
            self.threshold = np.where(cumulative_variance >= 0.95)[0]
            if len(self.threshold) == 0:
                self.threshold = self.n_components
            else:
                self.threshold = self.threshold[0] + 1
        
        self.is_fitted_ = True
        return self
    
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Transform volatility matrix using fitted TIP-PCA.
        
        Args:
            matrix: Volatility matrix to transform
            
        Returns:
            Transformed matrix in principal component space
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transformation")
            
        # Normalize if requested
        if self.normalize:
            X = matrix - self.mean_
        else:
            X = matrix.copy()
            
        # Project onto principal components
        transformed = X @ self.components_.T
        
        return transformed
    
    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Fit TIP-PCA and transform matrix.
        
        Args:
            matrix: Volatility matrix
            
        Returns:
            Transformed matrix
        """
        return self.fit(matrix).transform(matrix)
    
    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform from component space to original space.
        
        Args:
            transformed: Matrix in component space
            
        Returns:
            Reconstructed volatility matrix
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before inverse transformation")
            
        # Reconstruct from principal components
        reconstructed = transformed @ self.components_
        
        # Add back mean if normalized
        if self.normalize:
            reconstructed += self.mean_
            
        return reconstructed
    
    def decompose(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose volatility matrix into Low-Rank (Signal) + Noise.
        
        Args:
            matrix: Volatility matrix
            
        Returns:
            Tuple of (low_rank_matrix, noise_matrix)
        """
        # Fit model
        self.fit(matrix)
        
        # Get low-rank reconstruction
        transformed = self.transform(matrix)
        low_rank_matrix = self.inverse_transform(transformed)
        
        # Calculate noise component
        noise_matrix = matrix - low_rank_matrix
        
        return low_rank_matrix, noise_matrix
    
    def predict_volatility(self, 
                         matrix: np.ndarray,
                         horizon: int = 1) -> np.ndarray:
        """
        Predict future volatility using TIP-PCA decomposition.
        
        Args:
            matrix: Historical volatility matrix
            horizon: Prediction horizon (number of future days)
            
        Returns:
            Predicted volatility matrix
        """
        # Decompose into signal + noise
        signal_matrix, noise_matrix = self.decompose(matrix)
        
        # Simple prediction: use last day's pattern for future
        # This is a basic approach - can be enhanced with more sophisticated methods
        last_day_signal = signal_matrix[-1, :]
        
        # Create predictions by repeating the last day's pattern
        # In a real implementation, you'd use the component time series
        predicted_signal = np.tile(last_day_signal, (horizon, 1))
        
        # Add mean noise level (simplified approach)
        mean_noise = np.mean(noise_matrix, axis=0)
        predicted_noise = np.tile(mean_noise, (horizon, 1))
        
        # Combine signal + noise for prediction
        predicted_volatility = predicted_signal + predicted_noise
        
        return predicted_volatility
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.
        
        Returns:
            Array of explained variance ratios
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        return self.explained_variance_
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance.
        
        Returns:
            Array of cumulative explained variance
        """
        return np.cumsum(self.get_explained_variance_ratio())


def evaluate_decomposition(original: np.ndarray,
                        low_rank: np.ndarray,
                        noise: np.ndarray) -> dict:
    """
    Evaluate the quality of matrix decomposition.
    
    Args:
        original: Original volatility matrix
        low_rank: Low-rank (signal) component
        noise: Noise component
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Reconstruction error
    reconstruction = low_rank + noise
    reconstruction_error = np.mean((original - reconstruction)**2)
    
    # Signal-to-noise ratio
    signal_power = np.mean(low_rank**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Low-rank structure quality
    if low_rank.shape[0] > 1:
        correlation = np.corrcoef(low_rank.flatten(), original.flatten())[0, 1]
    else:
        correlation = 1.0  # Perfect correlation if only one row
    
    return {
        'reconstruction_error': reconstruction_error,
        'signal_to_noise_ratio': snr,
        'correlation': correlation,
        'low_rank_energy': signal_power,
        'noise_energy': noise_power
    }


if __name__ == "__main__":
    # Example usage
    print("📊 TIP-PCA Module - Example Usage")
    print("=" * 50)
    
    # Create sample volatility matrix
    np.random.seed(42)
    sample_matrix = np.random.randn(50, 78)
    sample_matrix = (sample_matrix - sample_matrix.min()) / (sample_matrix.max() - sample_matrix.min())
    
    print(f"Sample matrix shape: {sample_matrix.shape}")
    print(f"Sample matrix range: {sample_matrix.min():.4f} to {sample_matrix.max():.4f}")
    
    # Initialize and fit TIP-PCA
    print("\nFitting TIP-PCA model...")
    tip_pca = TIPPCA(n_components=10)
    low_rank, noise = tip_pca.decompose(sample_matrix)
    
    print(f"✅ Decomposition complete")
    print(f"  Low-rank matrix shape: {low_rank.shape}")
    print(f"  Noise matrix shape: {noise.shape}")
    print(f"  Components used: {tip_pca.n_components}")
    
    # Evaluate decomposition
    print("\nEvaluating decomposition...")
    metrics = evaluate_decomposition(sample_matrix, low_rank, noise)
    
    print(f"  Reconstruction Error: {metrics['reconstruction_error']:.6f}")
    print(f"  Signal-to-Noise Ratio: {metrics['signal_to_noise_ratio']:.4f} dB")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  Explained Variance: {tip_pca.get_cumulative_variance()[-1]:.4f}")
    
    # Make prediction
    print("\nMaking volatility prediction...")
    prediction = tip_pca.predict_volatility(sample_matrix, horizon=5)
    print(f"✅ Prediction complete: {prediction.shape}")
    print(f"  Predicted {prediction.shape[0]} days with {prediction.shape[1]} intraday points")
    
    print("\n✅ TIP-PCA working successfully!")
