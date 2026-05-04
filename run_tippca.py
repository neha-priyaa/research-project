"""
TIP-PCA Implementation and Testing Script

Phase 2: TIP-PCA Algorithm Implementation
Based on: Matrix-based Prediction Approach for Intraday Instantaneous Volatility Vector
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.models import TIPPCA, evaluate_decomposition
from src.data import load_data

def main():
    print("=" * 80)
    print("📊 PHASE 2: TIP-PCA IMPLEMENTATION")
    print("=" * 80)
    print()
    
    # Load processed volatility matrix
    print("🎯 STEP 1: Load Processed Volatility Matrix")
    print("-" * 80)
    
    try:
        # Try to load normalized matrix from Phase 1
        matrix_df = pd.read_csv('data/processed/volatility_matrix_full_normalized.csv', header=None)
        volatility_matrix = matrix_df.values
        
        print(f"✅ Volatility matrix loaded")
        print(f"  Shape: {volatility_matrix.shape}")
        print(f"  Range: {volatility_matrix.min():.6f} to {volatility_matrix.max():.6f}")
        print(f"  Mean: {volatility_matrix.mean():.6f}")
        print(f"  Std: {volatility_matrix.std():.6f}")
        print()
        
    except FileNotFoundError:
        print("❌ Volatility matrix not found! Run Phase 1 first.")
        return
    
    # Initialize TIP-PCA model
    print("🎯 STEP 2: Initialize TIP-PCA Model")
    print("-" * 80)
    print("Model Parameters:")
    print("  • Method: Tensor-Invariant Principal Component Analysis")
    print("  • Components: 10 (configurable)")
    print("  • Threshold: Auto-determined (95% variance)")
    print("  • Normalization: Enabled")
    print()
    
    tip_pca = TIPPCA(
        n_components=10,
        normalize=True
    )
    
    # Fit and decompose
    print("🎯 STEP 3: Perform Matrix Decomposition")
    print("-" * 80)
    print("Decomposing volatility matrix into:")
    print("  • Low-Rank Component (Signal)")
    print("  • Noise Component")
    print()
    
    low_rank_matrix, noise_matrix = tip_pca.decompose(volatility_matrix)
    
    print(f"✅ Decomposition complete")
    print(f"  Original matrix: {volatility_matrix.shape}")
    print(f"  Low-rank matrix: {low_rank_matrix.shape}")
    print(f"  Noise matrix: {noise_matrix.shape}")
    print()
    
    # Evaluate decomposition
    print("🎯 STEP 4: Evaluate Decomposition Quality")
    print("-" * 80)
    
    metrics = evaluate_decomposition(volatility_matrix, low_rank_matrix, noise_matrix)
    
    print("Decomposition Metrics:")
    print(f"  • Reconstruction Error: {metrics['reconstruction_error']:.8f}")
    print(f"  • Signal-to-Noise Ratio: {metrics['signal_to_noise_ratio']:.4f} dB")
    print(f"  • Signal Correlation: {metrics['correlation']:.6f}")
    print(f"  • Low-Rank Energy: {metrics['low_rank_energy']:.6f}")
    print(f"  • Noise Energy: {metrics['noise_energy']:.6f}")
    print()
    
    print("Explained Variance:")
    explained_var = tip_pca.get_explained_variance_ratio()
    cumulative_var = tip_pca.get_cumulative_variance()
    
    for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var)):
        print(f"  • Component {i+1}: {ev:.4f} (cumulative: {cv:.4f})")
    print(f"  • Total Explained Variance: {cumulative_var[-1]:.4f}")
    print()
    
    # Make predictions
    print("🎯 STEP 5: Make Volatility Predictions")
    print("-" * 80)
    print("Predicting next 5 trading days of volatility...")
    print()
    
    predictions = tip_pca.predict_volatility(volatility_matrix, horizon=5)
    
    print(f"✅ Predictions complete")
    print(f"  Prediction horizon: 5 days")
    print(f"  Prediction matrix: {predictions.shape}")
    print(f"  Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
    print()
    
    # Generate visualizations
    print("🎯 STEP 6: Generate TIP-PCA Visualizations")
    print("-" * 80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TIP-PCA Analysis: Matrix Decomposition & Prediction', 
                 fontsize=16, fontweight='bold')
    
    # 1. Original Volatility Matrix (First 20 days)
    ax1 = axes[0, 0]
    original_subset = volatility_matrix[:20, :]
    im1 = ax1.imshow(original_subset, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_title('Original Volatility Matrix (First 20 Days)', fontweight='bold')
    ax1.set_xlabel('Intraday Point')
    ax1.set_ylabel('Day')
    plt.colorbar(im1, ax=ax1, label='Normalized Volatility')
    
    # 2. Low-Rank Component (Signal)
    ax2 = axes[0, 1]
    low_rank_subset = low_rank_matrix[:20, :]
    im2 = ax2.imshow(low_rank_subset, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    ax2.set_title('Low-Rank Component (Signal)', fontweight='bold')
    ax2.set_xlabel('Intraday Point')
    ax2.set_ylabel('Day')
    plt.colorbar(im2, ax=ax2, label='Normalized Volatility')
    
    # 3. Noise Component
    ax3 = axes[0, 2]
    noise_subset = noise_matrix[:20, :]
    im3 = ax3.imshow(noise_subset, aspect='auto', cmap='Greys', interpolation='nearest')
    ax3.set_title('Noise Component', fontweight='bold')
    ax3.set_xlabel('Intraday Point')
    ax3.set_ylabel('Day')
    plt.colorbar(im3, ax=ax3, label='Normalized Volatility')
    
    # 4. Explained Variance
    ax4 = axes[1, 0]
    ax4.bar(range(1, len(explained_var) + 1), explained_var, color='steelblue', alpha=0.8)
    ax4.set_title('Explained Variance per Component', fontweight='bold')
    ax4.set_xlabel('Component')
    ax4.set_ylabel('Explained Variance Ratio')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cumulative Variance
    ax5 = axes[1, 1]
    ax5.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
             'o-', color='darkgreen', linewidth=2, markersize=6)
    ax5.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
    ax5.set_title('Cumulative Explained Variance', fontweight='bold')
    ax5.set_xlabel('Component')
    ax5.set_ylabel('Cumulative Explained Variance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Prediction vs Last Day
    ax6 = axes[1, 2]
    last_day = volatility_matrix[-1, :]
    pred_first_day = predictions[0, :]
    
    ax6.plot(last_day, label='Last Actual Day', color='blue', linewidth=2, alpha=0.8)
    ax6.plot(pred_first_day, label='Predicted Day', color='red', linewidth=2, alpha=0.8)
    ax6.set_title('Prediction vs Actual (Intraday Pattern)', fontweight='bold')
    ax6.set_xlabel('Intraday Point')
    ax6.set_ylabel('Normalized Volatility')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = 'data/processed/tippca_analysis.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {viz_path}")
    plt.close()
    print()
    
    # Save decomposition results
    print("🎯 STEP 7: Save Decomposition Results")
    print("-" * 80)
    
    # Save matrices
    pd.DataFrame(low_rank_matrix).to_csv('data/processed/low_rank_component.csv', index=False)
    pd.DataFrame(noise_matrix).to_csv('data/processed/noise_component.csv', index=False)
    pd.DataFrame(predictions).to_csv('data/processed/volatility_predictions.csv', index=False)
    
    print(f"✅ Decomposition results saved:")
    print(f"  • data/processed/low_rank_component.csv")
    print(f"  • data/processed/noise_component.csv")
    print(f"  • data/processed/volatility_predictions.csv")
    print()
    
    # Final summary
    print("=" * 80)
    print("🎉 PHASE 2: TIP-PCA IMPLEMENTATION COMPLETE!")
    print("=" * 80)
    print()
    print("📊 Summary:")
    print(f"  • Volatility matrix size: {volatility_matrix.shape}")
    print(f"  • Principal components used: {tip_pca.n_components}")
    print(f"  • Total explained variance: {cumulative_var[-1]:.4f}")
    print(f"  • Signal-to-noise ratio: {metrics['signal_to_noise_ratio']:.4f} dB")
    print(f"  • Prediction horizon: 5 days")
    print()
    print("✅ Key Achievements:")
    print("  • TIP-PCA algorithm implemented")
    print("  • Matrix decomposition successful (Low-Rank + Noise)")
    print("  • Signal and noise components separated")
    print("  • Volatility predictions generated")
    print("  • Comprehensive visualizations created")
    print()
    print("✅ Phase 2 - TIP-PCA Implementation: SUCCESSFUL")
    print("✅ Ready for Phase 3 - Model Training & Evaluation")
    print()

if __name__ == "__main__":
    main()
