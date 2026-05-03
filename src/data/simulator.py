"""
Data Simulator Module for Intraday Volatility Prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple

class DataSimulator:
    """Data simulator using Geometric Brownian Motion."""
    
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.2, drift: float = 0.05):
        self.initial_price = initial_price
        self.volatility = volatility
        self.drift = drift
        
    def simulate_gbm(self, days: int = 252, intraday_points: int = 390, dt: float = 1/252) -> np.ndarray:
        """Simulate price paths using Geometric Brownian Motion."""
        np.random.seed(42)
        daily_shocks = np.random.normal(0, 1, (days, intraday_points))
        intraday_dt = dt / intraday_points
        t = np.linspace(0, dt, intraday_points + 1)[:-1]
        
        prices = np.zeros((days, intraday_points))
        current_price = self.initial_price
        
        for day in range(days):
            mu = self.drift - 0.5 * self.volatility**2
            sigma = self.volatility
            price_path = current_price * np.exp(
                mu * t + sigma * np.sqrt(t) * daily_shocks[day]
            )
            prices[day] = price_path
            current_price = price_path[-1]
            
        return prices
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calculate log returns from price data."""
        returns = np.diff(np.log(prices), axis=1)
        return returns
    
    def calculate_volatility_matrix(self, prices: np.ndarray) -> np.ndarray:
        """Construct volatility matrix from price data."""
        returns = self.calculate_returns(prices)
        volatility_matrix = np.abs(returns)
        volatility_matrix = np.hstack([
            np.zeros((prices.shape[0], 1)),
            volatility_matrix
        ])
        return volatility_matrix
    
    def generate_dataset(self, days: int = 252, intraday_points: int = 390) -> pd.DataFrame:
        """Generate complete dataset with prices and volatility matrices."""
        prices = self.simulate_gbm(days, intraday_points)
        volatility = self.calculate_volatility_matrix(prices)
        
        data = []
        for day in range(days):
            for intraday in range(intraday_points):
                data.append({
                    'day': day,
                    'intraday': intraday,
                    'price': prices[day, intraday],
                    'volatility': volatility[day, intraday]
                })
        
        return pd.DataFrame(data)

def save_to_csv(data: pd.DataFrame, filepath: str) -> None:
    """Save simulated data to CSV file."""
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
