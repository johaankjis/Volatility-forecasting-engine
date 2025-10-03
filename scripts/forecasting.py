"""
Rolling Window Forecasting Framework
Implements out-of-sample forecasting with confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scripts.garch_model import GARCHModel
import warnings
warnings.filterwarnings('ignore')


class RollingWindowForecaster:
    """
    Rolling window forecasting framework for volatility models
    
    Implements expanding and rolling window schemes with proper
    out-of-sample evaluation
    """
    
    def __init__(self, returns: pd.Series, window_type: str = 'expanding',
                 window_size: int = 252):
        """
        Initialize rolling window forecaster
        
        Parameters:
        -----------
        returns : pd.Series
            Complete return series
        window_type : str
            'expanding' for expanding window, 'rolling' for fixed window
        window_size : int
            Initial window size (for expanding) or fixed size (for rolling)
        """
        self.returns = returns.dropna()
        self.window_type = window_type
        self.window_size = window_size
        self.forecasts = []
        self.actuals = []
        self.forecast_dates = []
        self.confidence_intervals = []
        
    def forecast_rolling(self, start_idx: Optional[int] = None,
                        end_idx: Optional[int] = None,
                        horizon: int = 1,
                        refit_frequency: int = 1,
                        p: int = 1, q: int = 1) -> pd.DataFrame:
        """
        Generate rolling window forecasts
        
        Parameters:
        -----------
        start_idx : int
            Starting index for forecasting (default: window_size)
        end_idx : int
            Ending index for forecasting (default: end of series)
        horizon : int
            Forecast horizon
        refit_frequency : int
            How often to refit model (1 = every period)
        p : int
            GARCH order
        q : int
            ARCH order
        
        Returns:
        --------
        pd.DataFrame : Forecast results with actuals and errors
        """
        if start_idx is None:
            start_idx = self.window_size
        if end_idx is None:
            end_idx = len(self.returns)
        
        print(f"\n{'='*60}")
        print(f"ROLLING WINDOW FORECASTING")
        print(f"{'='*60}")
        print(f"Window Type: {self.window_type}")
        print(f"Window Size: {self.window_size}")
        print(f"Forecast Horizon: {horizon}")
        print(f"Refit Frequency: {refit_frequency}")
        print(f"Model: GARCH({p},{q})")
        print(f"Forecasting periods: {start_idx} to {end_idx}")
        print(f"Total forecasts: {end_idx - start_idx}")
        print(f"{'='*60}\n")
        
        forecasts = []
        actuals = []
        dates = []
        lower_bounds = []
        upper_bounds = []
        
        fitted_model = None
        
        for t in range(start_idx, end_idx):
            # Determine training window
            if self.window_type == 'expanding':
                train_data = self.returns.iloc[:t]
            else:  # rolling
                train_data = self.returns.iloc[max(0, t - self.window_size):t]
            
            # Refit model if needed
            if fitted_model is None or (t - start_idx) % refit_frequency == 0:
                try:
                    model = GARCHModel(train_data, p=p, q=q)
                    model.fit(show_summary=False)
                    fitted_model = model
                except Exception as e:
                    print(f"Warning: Model fitting failed at t={t}: {str(e)}")
                    continue
            
            # Generate forecast
            try:
                forecast_df = fitted_model.forecast(horizon=horizon)
                forecast_vol = forecast_df['volatility'].iloc[horizon - 1]
                lower_ci = forecast_df['lower_95'].iloc[horizon - 1]
                upper_ci = forecast_df['upper_95'].iloc[horizon - 1]
            except Exception as e:
                print(f"Warning: Forecast failed at t={t}: {str(e)}")
                continue
            
            # Get actual realized volatility (using squared returns as proxy)
            if t + horizon - 1 < len(self.returns):
                actual_vol = np.abs(self.returns.iloc[t + horizon - 1])
                
                forecasts.append(forecast_vol)
                actuals.append(actual_vol)
                dates.append(self.returns.index[t + horizon - 1])
                lower_bounds.append(lower_ci)
                upper_bounds.append(upper_ci)
            
            # Progress indicator
            if (t - start_idx + 1) % 50 == 0:
                progress = ((t - start_idx + 1) / (end_idx - start_idx)) * 100
                print(f"Progress: {progress:.1f}% ({t - start_idx + 1}/{end_idx - start_idx} forecasts)")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'date': dates,
            'forecast': forecasts,
            'actual': actuals,
            'lower_95': lower_bounds,
            'upper_95': upper_bounds
        })
        
        # Calculate forecast errors
        results_df['error'] = results_df['actual'] - results_df['forecast']
        results_df['abs_error'] = np.abs(results_df['error'])
        results_df['squared_error'] = results_df['error'] ** 2
        results_df['pct_error'] = (results_df['error'] / results_df['actual']) * 100
        
        # Check if actual falls within confidence interval
        results_df['in_ci'] = (
            (results_df['actual'] >= results_df['lower_95']) &
            (results_df['actual'] <= results_df['upper_95'])
        )
        
        self.forecasts = forecasts
        self.actuals = actuals
        self.forecast_dates = dates
        
        print(f"\n{'='*60}")
        print(f"FORECASTING COMPLETE")
        print(f"{'='*60}")
        print(f"Total forecasts generated: {len(results_df)}")
        print(f"Coverage rate (95% CI): {results_df['in_ci'].mean() * 100:.2f}%")
        print(f"{'='*60}\n")
        
        return results_df
    
    def multi_horizon_forecast(self, horizons: List[int] = [1, 5, 10, 20],
                              start_idx: Optional[int] = None,
                              p: int = 1, q: int = 1) -> Dict[int, pd.DataFrame]:
        """
        Generate forecasts for multiple horizons
        
        Parameters:
        -----------
        horizons : List[int]
            List of forecast horizons
        start_idx : int
            Starting index for forecasting
        p : int
            GARCH order
        q : int
            ARCH order
        
        Returns:
        --------
        dict : Dictionary mapping horizon to forecast results
        """
        print(f"\n{'='*60}")
        print(f"MULTI-HORIZON FORECASTING")
        print(f"{'='*60}")
        print(f"Horizons: {horizons}")
        print(f"{'='*60}\n")
        
        results = {}
        
        for h in horizons:
            print(f"\nForecasting horizon: {h}")
            print("-" * 60)
            
            forecast_df = self.forecast_rolling(
                start_idx=start_idx,
                horizon=h,
                refit_frequency=5,  # Refit every 5 periods for efficiency
                p=p,
                q=q
            )
            
            results[h] = forecast_df
        
        return results


class VolatilityForecaster:
    """
    High-level interface for volatility forecasting
    """
    
    def __init__(self, returns: pd.Series):
        """
        Initialize volatility forecaster
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        """
        self.returns = returns
        self.model = None
        self.forecasts = None
        
    def fit_and_forecast(self, train_size: float = 0.8,
                        horizon: int = 1,
                        p: int = 1, q: int = 1) -> Tuple[GARCHModel, pd.DataFrame]:
        """
        Fit model on training data and forecast on test data
        
        Parameters:
        -----------
        train_size : float
            Proportion of data for training
        horizon : int
            Forecast horizon
        p : int
            GARCH order
        q : int
            ARCH order
        
        Returns:
        --------
        tuple : (fitted_model, forecast_results)
        """
        # Split data
        split_idx = int(len(self.returns) * train_size)
        train_data = self.returns.iloc[:split_idx]
        test_data = self.returns.iloc[split_idx:]
        
        print(f"\n{'='*60}")
        print(f"FIT AND FORECAST")
        print(f"{'='*60}")
        print(f"Total observations: {len(self.returns)}")
        print(f"Training set: {len(train_data)} ({train_size*100:.0f}%)")
        print(f"Test set: {len(test_data)} ({(1-train_size)*100:.0f}%)")
        print(f"{'='*60}\n")
        
        # Fit model on training data
        self.model = GARCHModel(train_data, p=p, q=q)
        self.model.fit(show_summary=True)
        
        # Run diagnostics
        self.model.diagnose()
        
        # Rolling forecast on test data
        forecaster = RollingWindowForecaster(
            self.returns,
            window_type='expanding',
            window_size=len(train_data)
        )
        
        self.forecasts = forecaster.forecast_rolling(
            start_idx=split_idx,
            horizon=horizon,
            refit_frequency=10,
            p=p,
            q=q
        )
        
        return self.model, self.forecasts
    
    def generate_forecast_path(self, horizon: int = 20) -> pd.DataFrame:
        """
        Generate multi-step ahead forecast path
        
        Parameters:
        -----------
        horizon : int
            Number of periods ahead
        
        Returns:
        --------
        pd.DataFrame : Forecast path with confidence bands
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        forecast_df = self.model.forecast(horizon=horizon)
        
        print(f"\n{'='*60}")
        print(f"FORECAST PATH ({horizon} periods ahead)")
        print(f"{'='*60}")
        print(forecast_df)
        print(f"{'='*60}\n")
        
        return forecast_df


# Example usage and testing
if __name__ == "__main__":
    print("Volatility Forecasting Engine - Forecasting Module")
    print("="*60)
    
    # Generate synthetic GARCH data
    np.random.seed(42)
    n = 500
    
    omega = 0.01
    alpha = 0.15
    beta = 0.80
    
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = volatility[t] * np.random.normal(0, 1)
    
    dates = pd.date_range(start='2022-01-01', periods=n, freq='D')
    returns_series = pd.Series(returns, index=dates, name='Returns')
    
    # Test 1: Simple fit and forecast
    print("\n" + "="*60)
    print("TEST 1: FIT AND FORECAST")
    print("="*60)
    
    forecaster = VolatilityForecaster(returns_series)
    model, forecast_results = forecaster.fit_and_forecast(
        train_size=0.7,
        horizon=1,
        p=1,
        q=1
    )
    
    print("\nForecast Results (first 10):")
    print(forecast_results.head(10))
    
    # Test 2: Multi-horizon forecasting
    print("\n" + "="*60)
    print("TEST 2: MULTI-HORIZON FORECASTING")
    print("="*60)
    
    rolling_forecaster = RollingWindowForecaster(
        returns_series,
        window_type='expanding',
        window_size=250
    )
    
    multi_horizon_results = rolling_forecaster.multi_horizon_forecast(
        horizons=[1, 5, 10],
        start_idx=300
    )
    
    print("\nMulti-horizon results summary:")
    for h, results in multi_horizon_results.items():
        print(f"\nHorizon {h}: {len(results)} forecasts")
        print(f"  Mean absolute error: {results['abs_error'].mean():.6f}")
        print(f"  Coverage rate: {results['in_ci'].mean() * 100:.2f}%")
    
    # Test 3: Forecast path
    print("\n" + "="*60)
    print("TEST 3: FORECAST PATH")
    print("="*60)
    
    forecast_path = forecaster.generate_forecast_path(horizon=20)
