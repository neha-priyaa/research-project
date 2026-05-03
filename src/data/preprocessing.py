"""
Data Preprocessing Module
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os

class DataProcessor:
    """Data processor for cleaning and preparing volatility data."""
    
    def __init__(self, threshold_std: float = 3.0, fill_method: str = 'forward_fill'):
        self.threshold_std = threshold_std
        self.fill_method = fill_method
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        data = self._fill_missing_values(data)
        data = self._remove_outliers(data)
        return data
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values based on specified method."""
        if self.fill_method == 'forward_fill':
            return data.ffill().bfill()
        elif self.fill_method == 'mean':
            return data.fillna(data.mean())
        elif self.fill_method == 'zero':
            return data.fillna(0)
        else:
            return data.dropna()
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using standard deviation threshold."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - self.threshold_std * std
            upper_bound = mean + self.threshold_std * std
            
            data.loc[data[col] < lower_bound, col] = lower_bound
            data.loc[data[col] > upper_bound, col] = upper_bound
            
        return data
    
    def construct_matrix(self, data: pd.DataFrame, day_col: str = 'day', 
                        intraday_col: str = 'intraday', value_col: str = 'volatility') -> np.ndarray:
        """Construct volatility matrix from DataFrame."""
        days = sorted(data[day_col].unique())
        intraday_periods = sorted(data[intraday_col].unique())
        
        n_days = len(days)
        n_intraday = len(intraday_periods)
        matrix = np.zeros((n_days, n_intraday))
        
        for i, day in enumerate(days):
            day_data = data[data[day_col] == day]
            for j, period in enumerate(intraday_periods):
                period_data = day_data[day_data[intraday_col] == period]
                if len(period_data) > 0:
                    matrix[i, j] = period_data[value_col].values[0]
                    
        return matrix
    
    def normalize_matrix(self, matrix: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize volatility matrix."""
        if method == 'minmax':
            min_val = matrix.min()
            max_val = matrix.max()
            return (matrix - min_val) / (max_val - min_val + 1e-8)
        elif method == 'zscore':
            mean = matrix.mean()
            std = matrix.std()
            return (matrix - mean) / (std + 1e-8)
        elif method == 'robust':
            median = np.median(matrix)
            q75 = np.percentile(matrix, 75)
            q25 = np.percentile(matrix, 25)
            iqr = q75 - q25
            return (matrix - median) / (iqr + 1e-8)
        return matrix

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = pd.read_csv(filepath)
    print(f"Loaded data from {filepath}")
    print(f"Shape: {data.shape}")
    return data

def process_volatility_matrix(data: pd.DataFrame, day_col: str = 'day',
                            intraday_col: str = 'intraday', 
                            value_col: str = 'volatility',
                            normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Process data and construct volatility matrix."""
    processor = DataProcessor()
    cleaned_data = processor.clean_data(data)
    raw_matrix = processor.construct_matrix(cleaned_data, day_col, intraday_col, value_col)
    
    if normalize:
        normalized_matrix = processor.normalize_matrix(raw_matrix, method='minmax')
    else:
        normalized_matrix = raw_matrix
    
    return raw_matrix, normalized_matrix

def save_matrix(matrix: np.ndarray, filepath: str, header: Optional = None) -> None:
    """Save volatility matrix to CSV file."""
    df = pd.DataFrame(matrix)
    if header is not None:
        df.columns = header
        
    df.to_csv(filepath, index=False)
    print(f"Matrix saved to {filepath}")
