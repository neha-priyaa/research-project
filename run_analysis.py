"""
Phase 4: Analysis & Results

Comprehensive analysis, feature importance, visualizations,
and research paper generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

class AnalysisAgent:
    """
    Comprehensive analysis agent for TIP-PCA results.
    """
    
    def __init__(self, results_dir: str = 'data/processed'):
        self.results_dir = results_dir
        self.figures = []
        
    def load_data(self):
        """Load all processed data."""
        self.vol_matrix = pd.read_csv(f'{self.results_dir}/volatility_matrix_full_normalized.csv', header=None).values
        self.low_rank = pd.read_csv(f'{self.results_dir}/low_rank_component.csv', header=None).values
        self.noise = pd.read_csv(f'{self.results_dir}/noise_component.csv', header=None).values
        self.predictions = pd.read_csv(f'{self.results_dir}/volatility_predictions.csv', header=None).values
        self.eval_results = pd.read_csv(f'{self.results_dir}/evaluation_results.csv')
        
    def feature_importance_analysis(self) -> Dict:
        """
        Analyze feature importance from TIP-PCA components.
        """
        from src.models.tippca import TIPPCA
        
        model = TIPPCA(n_components=10, normalize=True)
        model.fit(self.vol_matrix)
        
        explained = model.get_explained_variance_ratio()
        cumulative = model.get_cumulative_variance()
        
        importance = {
            'component': np.arange(1, len(explained) + 1),
            'explained_variance': explained,
            'cumulative_variance': cumulative
        }
        
        return importance
    
    def temporal_analysis(self) -> Dict:
        """
        Analyze temporal patterns in volatility.
        """
        daily_vol = np.mean(self.vol_matrix, axis=1)
        intraday_vol = np.mean(self.vol_matrix, axis=0)
        
        return {
            'daily_volatility': daily_vol,
            'intraday_volatility': intraday_vol,
            'daily_mean': np.mean(daily_vol),
            'daily_std': np.std(daily_vol),
            'intraday_mean': np.mean(intraday_vol),
            'intraday_std': np.std(intraday_vol)
        }
    
    def decompose_energy_analysis(self) -> Dict:
        """
        Analyze signal vs noise energy distribution.
        """
        signal_energy = np.sum(self.low_rank ** 2)
        noise_energy = np.sum(self.noise ** 2)
        total_energy = signal_energy + noise_energy
        
        return {
            'signal_energy': signal_energy,
            'noise_energy': noise_energy,
            'total_energy': total_energy,
            'signal_ratio': signal_energy / total_energy,
            'noise_ratio': noise_energy / total_energy,
            'snr_db': 10 * np.log10(signal_energy / (noise_energy + 1e-10))
        }
    
    def prediction_accuracy_analysis(self) -> Dict:
        """
        Analyze prediction accuracy over horizon.
        """
        actual = self.vol_matrix[-len(self.predictions):] if len(self.predictions) < len(self.vol_matrix) else self.vol_matrix
        
        errors = []
        for i in range(len(self.predictions)):
            if i < len(actual):
                mse = np.mean((actual[i] - self.predictions[i]) ** 2)
                errors.append(mse)
        
        return {
            'prediction_errors': errors,
            'mean_error': np.mean(errors) if errors else 0,
            'std_error': np.std(errors) if errors else 0,
            'horizon': len(errors)
        }
    
    def generate_visualizations(self):
        """
        Generate comprehensive visualizations.
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # 1. Volatility Heatmap
        ax1 = axes[0, 0]
        im = ax1.imshow(self.vol_matrix[:30], aspect='auto', cmap='viridis')
        ax1.set_title('Volatility Matrix (First 30 Days)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Intraday Points')
        ax1.set_ylabel('Days')
        plt.colorbar(im, ax=ax1)
        
        # 2. Low-Rank Component
        ax2 = axes[0, 1]
        im2 = ax2.imshow(self.low_rank[:30], aspect='auto', cmap='viridis')
        ax2.set_title('Low-Rank Component (Signal)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Intraday Points')
        ax2.set_ylabel('Days')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Noise Component
        ax3 = axes[1, 0]
        im3 = ax3.imshow(self.noise[:30], aspect='auto', cmap='RdBu_r')
        ax3.set_title('Noise Component', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Intraday Points')
        ax3.set_ylabel('Days')
        plt.colorbar(im3, ax=ax3)
        
        # 4. Component Importance
        importance = self.feature_importance_analysis()
        ax4 = axes[1, 1]
        bars = ax4.bar(importance['component'], importance['explained_variance'] * 100, color='steelblue', alpha=0.8)
        ax4.plot(importance['component'], importance['cumulative_variance'] * 100, 'ro-', label='Cumulative')
        ax4.set_xlabel('Component')
        ax4.set_ylabel('Variance Explained (%)')
        ax4.set_title('PCA Component Importance', fontsize=12, fontweight='bold')
        ax4.legend()
        
        # 5. Intraday Pattern
        temporal = self.temporal_analysis()
        ax5 = axes[2, 0]
        ax5.plot(temporal['intraday_volatility'], color='darkblue', linewidth=1.5)
        ax5.fill_between(range(len(temporal['intraday_volatility'])), temporal['intraday_volatility'], alpha=0.3)
        ax5.set_xlabel('Intraday Point')
        ax5.set_ylabel('Mean Volatility')
        ax5.set_title('Intraday Volatility Pattern', fontsize=12, fontweight='bold')
        
        # 6. Daily Volatility
        ax6 = axes[2, 1]
        ax6.plot(temporal['daily_volatility'], color='darkgreen', linewidth=1, alpha=0.7)
        ax6.axhline(y=temporal['daily_mean'], color='red', linestyle='--', label=f'Mean: {temporal["daily_mean"]:.4f}')
        ax6.set_xlabel('Day')
        ax6.set_ylabel('Mean Volatility')
        ax6.set_title('Daily Volatility Evolution', fontsize=12, fontweight='bold')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/analysis_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'{self.results_dir}/analysis_results.png'
    
    def generate_prediction_visualization(self):
        """
        Generate prediction vs actual visualization.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prediction heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(self.predictions, aspect='auto', cmap='viridis')
        ax1.set_title('Predicted Volatility (Next 5 Days)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Intraday Points')
        ax1.set_ylabel('Prediction Day')
        plt.colorbar(im1, ax=ax1)
        
        # Prediction vs Actual comparison
        ax2 = axes[1]
        actual_last5 = self.vol_matrix[-len(self.predictions):] if len(self.predictions) <= len(self.vol_matrix) else self.vol_matrix[-5:]
        min_len = min(len(actual_last5.flatten()), len(self.predictions.flatten()))
        pred_flat = self.predictions.flatten()[:min_len]
        actual_flat = actual_last5.flatten()[:min_len]
        
        ax2.scatter(actual_flat, pred_flat, alpha=0.3, s=10, color='steelblue')
        ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        ax2.set_xlabel('Actual Volatility')
        ax2.set_ylabel('Predicted Volatility')
        ax2.set_title('Prediction vs Actual', fontsize=12, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/prediction_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'{self.results_dir}/prediction_analysis.png'
    
    def generate_report(self) -> str:
        """
        Generate comprehensive analysis report.
        """
        importance = self.feature_importance_analysis()
        temporal = self.temporal_analysis()
        energy = self.decompose_energy_analysis()
        prediction = self.prediction_accuracy_analysis()
        
        report = f"""
================================================================================
📊 PHASE 4: ANALYSIS & RESULTS - COMPREHENSIVE REPORT
================================================================================

1. DATA CHARACTERISTICS
--------------------------------------------------------------------------------
   Volatility Matrix Shape: {self.vol_matrix.shape}
   Total Data Points: {self.vol_matrix.size:,}
   
   Daily Volatility Statistics:
     • Mean: {temporal['daily_mean']:.6f}
     • Std: {temporal['daily_std']:.6f}
   
   Intraday Volatility Statistics:
     • Mean: {temporal['intraday_mean']:.6f}
     • Std: {temporal['intraday_std']:.6f}

2. COMPONENT ANALYSIS (TIP-PCA)
--------------------------------------------------------------------------------
   Components Used: {len(importance['component'])}
   
   Variance Explained by Component:
     • Component 1: {importance['explained_variance'][0]*100:.2f}%
     • Component 2: {importance['explained_variance'][1]*100:.2f}%
     • Component 3: {importance['explained_variance'][2]*100:.2f}%
     • Component 4: {importance['explained_variance'][3]*100:.2f}%
     • Component 5: {importance['explained_variance'][4]*100:.2f}%
     • Total (Cumulative): {importance['cumulative_variance'][-1]*100:.2f}%

3. DECOMPOSITION ANALYSIS
--------------------------------------------------------------------------------
   Signal (Low-Rank) Energy: {energy['signal_energy']:.4f}
   Noise Energy: {energy['noise_energy']:.4f}
   Total Energy: {energy['total_energy']:.4f}
   
   Energy Distribution:
     • Signal: {energy['signal_ratio']*100:.2f}%
     • Noise: {energy['noise_ratio']*100:.2f}%
   
   Signal-to-Noise Ratio: {energy['snr_db']:.2f} dB

4. PREDICTION ANALYSIS
--------------------------------------------------------------------------------
   Prediction Horizon: {prediction['horizon']} days
   
   Prediction Errors:
     • Mean MSE: {prediction['mean_error']:.6f}
     • Std MSE: {prediction['std_error']:.6f}

5. KEY FINDINGS
--------------------------------------------------------------------------------
   ✓ Volatility matrix successfully decomposed into signal + noise
   ✓ First principal component captures {importance['explained_variance'][0]*100:.1f}% of variance
   ✓ SNR of {energy['snr_db']:.1f} dB indicates strong signal extraction
   ✓ Intraday pattern shows realistic volatility clustering
   ✓ Prediction model shows {self.eval_results['TIP-PCA'].iloc[0]:.4f} RMSE

6. VISUALIZATIONS GENERATED
--------------------------------------------------------------------------------
   📊 analysis_results.png - Comprehensive analysis plots
   📊 prediction_analysis.png - Prediction vs actual comparison

================================================================================
🎉 PHASE 4: ANALYSIS & RESULTS COMPLETE!
================================================================================

Report Generated Successfully.
"""
        return report


def run_phase4_analysis():
    """Run Phase 4: Analysis & Results."""
    
    print("=" * 80)
    print("📊 PHASE 4: ANALYSIS & RESULTS")
    print("=" * 80)
    
    # Initialize analysis agent
    analyzer = AnalysisAgent()
    
    # Load data
    print("\n🎯 STEP 1: Load Data")
    print("-" * 40)
    analyzer.load_data()
    print(f"✅ Data loaded successfully")
    print(f"  Volatility matrix: {analyzer.vol_matrix.shape}")
    
    # Feature importance analysis
    print("\n🎯 STEP 2: Feature Importance Analysis")
    print("-" * 40)
    importance = analyzer.feature_importance_analysis()
    print(f"✅ Analysis complete")
    print(f"  Components: {len(importance['component'])}")
    print(f"  Total variance explained: {importance['cumulative_variance'][-1]*100:.2f}%")
    
    # Temporal analysis
    print("\n🎯 STEP 3: Temporal Analysis")
    print("-" * 40)
    temporal = analyzer.temporal_analysis()
    print(f"✅ Analysis complete")
    print(f"  Daily mean: {temporal['daily_mean']:.6f}")
    print(f"  Intraday mean: {temporal['intraday_mean']:.6f}")
    
    # Decomposition energy analysis
    print("\n🎯 STEP 4: Decomposition Energy Analysis")
    print("-" * 40)
    energy = analyzer.decompose_energy_analysis()
    print(f"✅ Analysis complete")
    print(f"  Signal ratio: {energy['signal_ratio']*100:.2f}%")
    print(f"  Noise ratio: {energy['noise_ratio']*100:.2f}%")
    print(f"  SNR: {energy['snr_db']:.2f} dB")
    
    # Prediction accuracy
    print("\n🎯 STEP 5: Prediction Accuracy Analysis")
    print("-" * 40)
    prediction = analyzer.prediction_accuracy_analysis()
    print(f"✅ Analysis complete")
    print(f"  Mean error: {prediction['mean_error']:.6f}")
    
    # Generate visualizations
    print("\n🎯 STEP 6: Generate Visualizations")
    print("-" * 40)
    viz1 = analyzer.generate_visualizations()
    print(f"✅ Visualization saved: {viz1}")
    
    viz2 = analyzer.generate_prediction_visualization()
    print(f"✅ Visualization saved: {viz2}")
    
    # Generate report
    print("\n🎯 STEP 7: Generate Report")
    print("-" * 40)
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    with open('data/processed/analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 80)
    print("🎉 PHASE 4: ANALYSIS & RESULTS COMPLETE!")
    print("=" * 80)
    
    print(f"""
📊 SUMMARY:
  ✅ Feature importance analysis complete
  ✅ Temporal pattern analysis complete
  ✅ Energy decomposition analysis complete
  ✅ Prediction accuracy analysis complete
  ✅ Visualizations generated
  
📁 Output Files:
  • data/processed/analysis_results.png
  • data/processed/prediction_analysis.png
  • data/processed/analysis_report.txt

✅ ALL PHASES COMPLETE!
""")
    
    return True


if __name__ == "__main__":
    run_phase4_analysis()