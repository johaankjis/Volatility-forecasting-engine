"""
Evaluation Metrics and Visualization
Comprehensive forecast evaluation with statistical tests and plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ForecastEvaluator:
    """
    Comprehensive evaluation of volatility forecasts
    """
    
    def __init__(self, forecasts: pd.Series, actuals: pd.Series):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        forecasts : pd.Series
            Forecasted values
        actuals : pd.Series
            Actual realized values
        """
        self.forecasts = forecasts
        self.actuals = actuals
        self.errors = actuals - forecasts
        self.metrics = {}
        
    def calculate_mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error
        
        Returns:
        --------
        float : MAPE value
        """
        mape = np.mean(np.abs(self.errors / self.actuals)) * 100
        self.metrics['mape'] = mape
        return mape
    
    def calculate_mae(self) -> float:
        """
        Calculate Mean Absolute Error
        
        Returns:
        --------
        float : MAE value
        """
        mae = np.mean(np.abs(self.errors))
        self.metrics['mae'] = mae
        return mae
    
    def calculate_rmse(self) -> float:
        """
        Calculate Root Mean Squared Error
        
        Returns:
        --------
        float : RMSE value
        """
        rmse = np.sqrt(np.mean(self.errors ** 2))
        self.metrics['rmse'] = rmse
        return rmse
    
    def calculate_mse(self) -> float:
        """
        Calculate Mean Squared Error
        
        Returns:
        --------
        float : MSE value
        """
        mse = np.mean(self.errors ** 2)
        self.metrics['mse'] = mse
        return mse
    
    def calculate_qlike(self) -> float:
        """
        Calculate QLIKE (Quasi-Likelihood) loss function
        Robust measure for volatility forecasting
        
        QLIKE = log(forecast) + actual²/forecast
        
        Returns:
        --------
        float : QLIKE value
        """
        qlike = np.mean(np.log(self.forecasts ** 2) + (self.actuals ** 2) / (self.forecasts ** 2))
        self.metrics['qlike'] = qlike
        return qlike
    
    def calculate_r2(self) -> float:
        """
        Calculate R-squared
        
        Returns:
        --------
        float : R² value
        """
        ss_res = np.sum(self.errors ** 2)
        ss_tot = np.sum((self.actuals - np.mean(self.actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        self.metrics['r2'] = r2
        return r2
    
    def mincer_zarnowitz_test(self) -> Dict:
        """
        Mincer-Zarnowitz regression test for forecast unbiasedness
        
        Regression: actual = α + β·forecast + ε
        H0: α=0 and β=1 (unbiased forecast)
        
        Returns:
        --------
        dict : Test results
        """
        from scipy.stats import t as t_dist
        
        # Run regression
        X = np.column_stack([np.ones(len(self.forecasts)), self.forecasts])
        y = self.actuals.values
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        residuals = y - y_pred
        
        # Standard errors
        n = len(y)
        k = 2
        sigma2 = np.sum(residuals ** 2) / (n - k)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # T-statistics
        t_alpha = beta[0] / se_beta[0]
        t_beta = (beta[1] - 1) / se_beta[1]
        
        # P-values
        p_alpha = 2 * (1 - t_dist.cdf(np.abs(t_alpha), n - k))
        p_beta = 2 * (1 - t_dist.cdf(np.abs(t_beta), n - k))
        
        results = {
            'alpha': beta[0],
            'beta': beta[1],
            'se_alpha': se_beta[0],
            'se_beta': se_beta[1],
            't_alpha': t_alpha,
            't_beta': t_beta,
            'p_alpha': p_alpha,
            'p_beta': p_beta,
            'unbiased': (p_alpha > 0.05) and (p_beta > 0.05)
        }
        
        self.metrics['mincer_zarnowitz'] = results
        return results
    
    def diebold_mariano_test(self, alternative_forecasts: pd.Series,
                            loss_fn: str = 'mse') -> Dict:
        """
        Diebold-Mariano test for comparing forecast accuracy
        
        Parameters:
        -----------
        alternative_forecasts : pd.Series
            Alternative model forecasts
        loss_fn : str
            Loss function ('mse' or 'mae')
        
        Returns:
        --------
        dict : Test results
        """
        # Calculate loss differential
        if loss_fn == 'mse':
            loss1 = (self.actuals - self.forecasts) ** 2
            loss2 = (self.actuals - alternative_forecasts) ** 2
        elif loss_fn == 'mae':
            loss1 = np.abs(self.actuals - self.forecasts)
            loss2 = np.abs(self.actuals - alternative_forecasts)
        else:
            raise ValueError("loss_fn must be 'mse' or 'mae'")
        
        d = loss1 - loss2
        
        # Calculate DM statistic
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        n = len(d)
        
        dm_stat = d_mean / np.sqrt(d_var / n)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        results = {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'loss_differential': d_mean,
            'model1_better': dm_stat < 0 and p_value < 0.05
        }
        
        return results
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all evaluation metrics
        
        Returns:
        --------
        dict : All metrics
        """
        print(f"\n{'='*60}")
        print(f"FORECAST EVALUATION METRICS")
        print(f"{'='*60}")
        
        # Error metrics
        mae = self.calculate_mae()
        mse = self.calculate_mse()
        rmse = self.calculate_rmse()
        mape = self.calculate_mape()
        qlike = self.calculate_qlike()
        r2 = self.calculate_r2()
        
        print(f"\nError Metrics:")
        print(f"  MAE:   {mae:.6f}")
        print(f"  MSE:   {mse:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAPE:  {mape:.2f}%")
        print(f"  QLIKE: {qlike:.6f}")
        print(f"  R²:    {r2:.4f}")
        
        # Mincer-Zarnowitz test
        mz_results = self.mincer_zarnowitz_test()
        print(f"\nMincer-Zarnowitz Test:")
        print(f"  α (intercept): {mz_results['alpha']:.6f} (p={mz_results['p_alpha']:.4f})")
        print(f"  β (slope):     {mz_results['beta']:.6f} (p={mz_results['p_beta']:.4f})")
        print(f"  → {'✓ Unbiased' if mz_results['unbiased'] else '✗ Biased'}")
        
        # Error statistics
        print(f"\nError Statistics:")
        print(f"  Mean Error:    {np.mean(self.errors):.6f}")
        print(f"  Std Error:     {np.std(self.errors):.6f}")
        print(f"  Min Error:     {np.min(self.errors):.6f}")
        print(f"  Max Error:     {np.max(self.errors):.6f}")
        
        print(f"{'='*60}\n")
        
        return self.metrics


class VolatilityVisualizer:
    """
    Visualization tools for volatility forecasting
    """
    
    @staticmethod
    def plot_volatility_series(returns: pd.Series, conditional_vol: pd.Series,
                               title: str = "Conditional Volatility",
                               save_path: Optional[str] = None):
        """
        Plot returns and conditional volatility
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        conditional_vol : pd.Series
            Conditional volatility series
        title : str
            Plot title
        save_path : str
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot returns
        ax1.plot(returns.index, returns.values, linewidth=0.5, color='steelblue', alpha=0.7)
        ax1.set_title(f'{title} - Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Returns', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Plot conditional volatility
        ax2.plot(conditional_vol.index, conditional_vol.values, 
                linewidth=1.5, color='darkred', label='Conditional Volatility')
        ax2.fill_between(conditional_vol.index, 0, conditional_vol.values, 
                         alpha=0.3, color='darkred')
        ax2.set_title(f'{title} - Conditional Volatility', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volatility', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_forecast_vs_actual(forecast_df: pd.DataFrame,
                               title: str = "Forecast vs Actual",
                               save_path: Optional[str] = None):
        """
        Plot forecasts against actual values with confidence intervals
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            DataFrame with 'forecast', 'actual', 'lower_95', 'upper_95' columns
        title : str
            Plot title
        save_path : str
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        dates = forecast_df['date'] if 'date' in forecast_df.columns else forecast_df.index
        
        # Plot actual values
        ax.plot(dates, forecast_df['actual'], 
               linewidth=2, color='black', label='Actual', alpha=0.8)
        
        # Plot forecasts
        ax.plot(dates, forecast_df['forecast'], 
               linewidth=2, color='steelblue', label='Forecast', linestyle='--')
        
        # Plot confidence intervals
        ax.fill_between(dates, 
                       forecast_df['lower_95'], 
                       forecast_df['upper_95'],
                       alpha=0.2, color='steelblue', label='95% CI')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Volatility', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_forecast_errors(forecast_df: pd.DataFrame,
                            title: str = "Forecast Errors",
                            save_path: Optional[str] = None):
        """
        Plot forecast error analysis
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            DataFrame with error columns
        title : str
            Plot title
        save_path : str
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        dates = forecast_df['date'] if 'date' in forecast_df.columns else forecast_df.index
        
        # Error over time
        axes[0, 0].plot(dates, forecast_df['error'], linewidth=1, color='darkred')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[0, 0].set_title('Forecast Errors Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[0, 1].hist(forecast_df['error'], bins=50, color='steelblue', 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Error Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot: Forecast vs Actual
        axes[1, 0].scatter(forecast_df['forecast'], forecast_df['actual'], 
                          alpha=0.5, color='steelblue', s=20)
        
        # Add 45-degree line
        min_val = min(forecast_df['forecast'].min(), forecast_df['actual'].min())
        max_val = max(forecast_df['forecast'].max(), forecast_df['actual'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='Perfect Forecast')
        
        axes[1, 0].set_title('Forecast vs Actual', fontweight='bold')
        axes[1, 0].set_xlabel('Forecast')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(forecast_df['error'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Error Normality)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_multi_horizon_comparison(multi_horizon_results: Dict[int, pd.DataFrame],
                                     metric: str = 'mae',
                                     save_path: Optional[str] = None):
        """
        Compare forecast accuracy across multiple horizons
        
        Parameters:
        -----------
        multi_horizon_results : dict
            Dictionary mapping horizon to forecast results
        metric : str
            Metric to compare ('mae', 'rmse', 'mape')
        save_path : str
            Path to save figure
        """
        horizons = sorted(multi_horizon_results.keys())
        
        if metric == 'mae':
            values = [multi_horizon_results[h]['abs_error'].mean() for h in horizons]
            ylabel = 'Mean Absolute Error'
        elif metric == 'rmse':
            values = [np.sqrt((multi_horizon_results[h]['squared_error']).mean()) for h in horizons]
            ylabel = 'Root Mean Squared Error'
        elif metric == 'mape':
            values = [multi_horizon_results[h]['pct_error'].abs().mean() for h in horizons]
            ylabel = 'Mean Absolute Percentage Error (%)'
        else:
            raise ValueError("metric must be 'mae', 'rmse', or 'mape'")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(horizons, values, marker='o', linewidth=2, markersize=8, 
               color='steelblue', label=ylabel)
        ax.fill_between(horizons, values, alpha=0.3, color='steelblue')
        
        ax.set_title(f'Forecast Accuracy by Horizon', fontsize=14, fontweight='bold')
        ax.set_xlabel('Forecast Horizon', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Volatility Forecasting Engine - Evaluation Module")
    print("="*60)
    
    # Generate synthetic forecast data
    np.random.seed(42)
    n = 200
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    actuals = np.abs(np.random.normal(0.02, 0.01, n))
    forecasts = actuals + np.random.normal(0, 0.005, n)
    
    forecast_df = pd.DataFrame({
        'date': dates,
        'forecast': forecasts,
        'actual': actuals,
        'lower_95': forecasts * 0.7,
        'upper_95': forecasts * 1.3
    })
    
    forecast_df['error'] = forecast_df['actual'] - forecast_df['forecast']
    forecast_df['abs_error'] = np.abs(forecast_df['error'])
    forecast_df['squared_error'] = forecast_df['error'] ** 2
    forecast_df['pct_error'] = (forecast_df['error'] / forecast_df['actual']) * 100
    
    # Evaluate forecasts
    evaluator = ForecastEvaluator(
        forecasts=pd.Series(forecasts),
        actuals=pd.Series(actuals)
    )
    
    metrics = evaluator.calculate_all_metrics()
    
    # Visualize
    print("\nGenerating visualizations...")
    
    visualizer = VolatilityVisualizer()
    
    # Plot forecast vs actual
    visualizer.plot_forecast_vs_actual(forecast_df, title="Volatility Forecast Evaluation")
    
    # Plot forecast errors
    visualizer.plot_forecast_errors(forecast_df, title="Forecast Error Analysis")
    
    print("\n✓ Evaluation complete!")
