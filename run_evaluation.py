"""
Phase 3: Model Training & Evaluation

Comprehensive evaluation of TIP-PCA model with train/test split,
cross-validation, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluator for TIP-PCA volatility predictions.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.metrics_ = {}
        
    def train_test_split(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        np.random.seed(self.random_state)
        n_samples = matrix.shape[0]
        n_test = int(n_samples * self.test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_set = matrix[train_indices]
        test_set = matrix[test_indices]
        
        return train_set, test_set
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))
    
    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate all evaluation metrics."""
        return {
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAE': self.calculate_mae(y_true, y_pred),
            'MSE': self.calculate_mse(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'R2': self.calculate_r2(y_true, y_pred)
        }


def run_phase3_evaluation():
    """Run Phase 3: Model Training & Evaluation."""
    
    print("=" * 80)
    print("📊 PHASE 3: MODEL TRAINING & EVALUATION")
    print("=" * 80)
    
    # Load volatility matrix
    print("\n🎯 STEP 1: Load Volatility Matrix")
    print("-" * 40)
    vol_matrix = pd.read_csv('data/processed/volatility_matrix_full_normalized.csv', header=None).values
    print(f"✅ Volatility matrix loaded")
    print(f"  Shape: {vol_matrix.shape}")
    print(f"  Range: {vol_matrix.min():.6f} to {vol_matrix.max():.6f}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(test_size=0.2, random_state=42)
    
    # Train/test split
    print("\n🎯 STEP 2: Train/Test Split")
    print("-" * 40)
    train_matrix, test_matrix = evaluator.train_test_split(vol_matrix)
    print(f"✅ Data split complete")
    print(f"  Training set: {train_matrix.shape}")
    print(f"  Test set: {test_matrix.shape}")
    print(f"  Split ratio: 80/20")
    
    # Import TIP-PCA model
    from src.models.tippca import TIPPCA
    
    # Train model on training data
    print("\n🎯 STEP 3: Train TIP-PCA Model")
    print("-" * 40)
    model = TIPPCA(n_components=10, normalize=True)
    model.fit(train_matrix)
    print(f"✅ Model trained")
    print(f"  Components: {model.n_components}")
    print(f"  Explained Variance: {model.get_cumulative_variance()[-1]:.4f}")
    
    # Decompose training data
    train_low_rank, train_noise = model.decompose(train_matrix)
    
    # Generate predictions for test set
    print("\n🎯 STEP 4: Generate Predictions")
    print("-" * 40)
    predictions = model.predict_volatility(test_matrix, horizon=len(test_matrix))
    print(f"✅ Predictions generated")
    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
    
    # Evaluate model
    print("\n🎯 STEP 5: Calculate Evaluation Metrics")
    print("-" * 40)
    metrics = evaluator.calculate_all_metrics(test_matrix.flatten(), predictions.flatten())
    
    print(f"\n📈 EVALUATION METRICS:")
    print(f"  ┌{'─' * 30}┐")
    for metric_name, value in metrics.items():
        unit = '%' if metric_name == 'MAPE' else ''
        print(f"  │ {metric_name:12}: {value:>10.6f}{unit} │")
    print(f"  └{'─' * 30}┘")
    
    # Cross-validation
    print("\n🎯 STEP 6: Cross-Validation (5-Fold)")
    print("-" * 40)
    cv_scores = []
    n_folds = 5
    fold_size = len(vol_matrix) // n_folds
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        cv_train = np.vstack([vol_matrix[:val_start], vol_matrix[val_end:]])
        cv_val = vol_matrix[val_start:val_end]
        
        cv_model = TIPPCA(n_components=10, normalize=True)
        cv_model.fit(cv_train)
        cv_pred = cv_model.predict_volatility(cv_val, horizon=len(cv_val))
        
        fold_rmse = evaluator.calculate_rmse(cv_val, cv_pred)
        cv_scores.append(fold_rmse)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"✅ Cross-validation complete")
    print(f"  Fold scores (RMSE): {[f'{s:.6f}' for s in cv_scores]}")
    print(f"  Mean RMSE: {cv_mean:.6f}")
    print(f"  Std RMSE: {cv_std:.6f}")
    
    # Baseline comparison
    print("\n🎯 STEP 7: Baseline Model Comparison")
    print("-" * 40)
    
    # Baseline 1: Mean prediction
    baseline_mean = np.full_like(test_matrix, train_matrix.mean())
    baseline_metrics = evaluator.calculate_all_metrics(test_matrix, baseline_mean)
    
    # Baseline 2: Last day prediction
    baseline_last = np.tile(train_matrix[-1], (len(test_matrix), 1))
    last_metrics = evaluator.calculate_all_metrics(test_matrix, baseline_last)
    
    print(f"📊 BASELINE COMPARISON:")
    print(f"\n  TIP-PCA Model:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}")
    
    print(f"\n  Baseline (Mean):")
    for k, v in baseline_metrics.items():
        print(f"    {k}: {v:.6f}")
    
    print(f"\n  Baseline (Last Day):")
    for k, v in last_metrics.items():
        print(f"    {k}: {v:.6f}")
    
    # Performance summary
    print("\n🎯 STEP 8: Performance Summary")
    print("-" * 40)
    
    improvement_vs_mean = ((baseline_metrics['RMSE'] - metrics['RMSE']) / baseline_metrics['RMSE']) * 100
    improvement_vs_last = ((last_metrics['RMSE'] - metrics['RMSE']) / last_metrics['RMSE']) * 100
    
    print(f"✅ TIP-PCA vs Baseline improvements:")
    print(f"  vs Mean prediction: {improvement_vs_mean:+.2f}%")
    print(f"  vs Last day: {improvement_vs_last:+.2f}%")
    
    # Save results
    print("\n🎯 STEP 9: Save Results")
    print("-" * 40)
    
    results_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'TIP-PCA': list(metrics.values()),
        'Baseline_Mean': list(baseline_metrics.values()),
        'Baseline_Last': list(last_metrics.values())
    })
    results_df.to_csv('data/processed/evaluation_results.csv', index=False)
    
    cv_results = pd.DataFrame({
        'Fold': range(1, n_folds + 1),
        'RMSE': cv_scores
    })
    cv_results.to_csv('data/processed/cv_results.csv', index=False)
    
    print(f"✅ Results saved:")
    print(f"  • data/processed/evaluation_results.csv")
    print(f"  • data/processed/cv_results.csv")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 PHASE 3: MODEL TRAINING & EVALUATION COMPLETE!")
    print("=" * 80)
    
    print(f"""
📊 SUMMARY:
  • Training samples: {train_matrix.shape[0]}
  • Test samples: {test_matrix.shape[0]}
  • Components: {model.n_components}
  • Total Explained Variance: {model.get_cumulative_variance()[-1]:.4f}
  
📈 KEY METRICS:
  • RMSE: {metrics['RMSE']:.6f}
  • MAE: {metrics['MAE']:.6f}
  • R²: {metrics['R2']:.6f}
  
📊 CROSS-VALIDATION:
  • Mean RMSE: {cv_mean:.6f} ± {cv_std:.6f}
  
⚡ IMPROVEMENT:
  • vs Mean baseline: {improvement_vs_mean:+.2f}%
  • vs Last day baseline: {improvement_vs_last:+.2f}%

✅ Phase 3 - Model Training & Evaluation: SUCCESSFUL
✅ Ready for Phase 4 - Analysis & Results
""")
    
    return metrics, cv_scores


if __name__ == "__main__":
    run_phase3_evaluation()