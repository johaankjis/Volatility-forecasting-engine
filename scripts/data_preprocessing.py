"""
Data Preprocessing & Stationarity Checks
Handles cleaning, transformation, and statistical testing of financial time series
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive preprocessing pipeline for financial time series data
    """
    
    def __init__(self, data: pd.Series, name: str = "Asset"):
        """
        Initialize preprocessor with price data
        
        Parameters:
        -----------
        data : pd.Series
            Time series of asset prices (indexed by date)
        name : str
            Name of the asset for reporting
        """
        self.raw_data = data
        self.name = name
        self.returns = None
        self.log_returns = None
        self.stationary_series = None
        self.preprocessing_report = {}
        
    def compute_returns(self, method: str = 'log') -> pd.Series:
        """
        Compute returns from price series
        
        Parameters:
        -----------
        method : str
            'log' for log returns, 'simple' for arithmetic returns
        
        Returns:
        --------
        pd.Series : Computed returns
        """
        if method == 'log':
            self.log_returns = np.log(self.raw_data / self.raw_data.shift(1))
            self.returns = self.log_returns
        elif method == 'simple':
            self.returns = self.raw_data.pct_change()
        else:
            raise ValueError("Method must be 'log' or 'simple'")
        
        # Remove NaN values
        self.returns = self.returns.dropna()
        
        print(f"✓ Computed {method} returns: {len(self.returns)} observations")
        return self.returns
    
    def check_stationarity(self, series: pd.Series = None, alpha: float = 0.05) -> dict:
        """
        Perform comprehensive stationarity tests
        
        Parameters:
        -----------
        series : pd.Series
            Series to test (defaults to returns)
        alpha : float
            Significance level for tests
        
        Returns:
        --------
        dict : Test results and interpretation
        """
        if series is None:
            series = self.returns
        
        # Augmented Dickey-Fuller Test
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_statistic, adf_pvalue = adf_result[0], adf_result[1]
        adf_critical = adf_result[4]
        
        # KPSS Test (null hypothesis: series is stationary)
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        kpss_statistic, kpss_pvalue = kpss_result[0], kpss_result[1]
        kpss_critical = kpss_result[3]
        
        # Interpretation
        is_stationary_adf = adf_pvalue < alpha
        is_stationary_kpss = kpss_pvalue > alpha
        
        results = {
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'adf_critical_values': adf_critical,
            'adf_stationary': is_stationary_adf,
            'kpss_statistic': kpss_statistic,
            'kpss_pvalue': kpss_pvalue,
            'kpss_critical_values': kpss_critical,
            'kpss_stationary': is_stationary_kpss,
            'consensus_stationary': is_stationary_adf and is_stationary_kpss
        }
        
        self.preprocessing_report['stationarity'] = results
        
        # Print results
        print(f"\n{'='*60}")
        print(f"STATIONARITY TESTS: {self.name}")
        print(f"{'='*60}")
        print(f"ADF Test:")
        print(f"  Statistic: {adf_statistic:.4f}")
        print(f"  P-value: {adf_pvalue:.4f}")
        print(f"  Critical Values: {adf_critical}")
        print(f"  → {'STATIONARY' if is_stationary_adf else 'NON-STATIONARY'} (reject H0: unit root)")
        print(f"\nKPSS Test:")
        print(f"  Statistic: {kpss_statistic:.4f}")
        print(f"  P-value: {kpss_pvalue:.4f}")
        print(f"  Critical Values: {kpss_critical}")
        print(f"  → {'STATIONARY' if is_stationary_kpss else 'NON-STATIONARY'} (fail to reject H0: stationary)")
        print(f"\n{'✓' if results['consensus_stationary'] else '✗'} Consensus: {'STATIONARY' if results['consensus_stationary'] else 'NON-STATIONARY'}")
        print(f"{'='*60}\n")
        
        return results
    
    def apply_differencing(self, order: int = 1) -> pd.Series:
        """
        Apply differencing to achieve stationarity
        
        Parameters:
        -----------
        order : int
            Order of differencing
        
        Returns:
        --------
        pd.Series : Differenced series
        """
        series = self.returns.copy()
        for i in range(order):
            series = series.diff()
        
        self.stationary_series = series.dropna()
        print(f"✓ Applied differencing (order={order}): {len(self.stationary_series)} observations")
        
        return self.stationary_series
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> dict:
        """
        Detect outliers in returns series
        
        Parameters:
        -----------
        method : str
            'iqr' for interquartile range, 'zscore' for standard deviation
        threshold : float
            Threshold for outlier detection
        
        Returns:
        --------
        dict : Outlier statistics and indices
        """
        series = self.returns.dropna()
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outliers = z_scores > threshold
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        outlier_indices = series[outliers].index
        outlier_values = series[outliers].values
        
        results = {
            'method': method,
            'threshold': threshold,
            'n_outliers': outliers.sum(),
            'outlier_pct': (outliers.sum() / len(series)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values
        }
        
        self.preprocessing_report['outliers'] = results
        
        print(f"\n{'='*60}")
        print(f"OUTLIER DETECTION: {self.name}")
        print(f"{'='*60}")
        print(f"Method: {method.upper()} (threshold={threshold})")
        print(f"Outliers detected: {results['n_outliers']} ({results['outlier_pct']:.2f}%)")
        print(f"{'='*60}\n")
        
        return results
    
    def compute_descriptive_stats(self) -> dict:
        """
        Compute comprehensive descriptive statistics
        
        Returns:
        --------
        dict : Descriptive statistics
        """
        series = self.returns.dropna()
        
        stats_dict = {
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'jarque_bera': stats.jarque_bera(series),
            'median': series.median(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75)
        }
        
        self.preprocessing_report['descriptive_stats'] = stats_dict
        
        print(f"\n{'='*60}")
        print(f"DESCRIPTIVE STATISTICS: {self.name}")
        print(f"{'='*60}")
        print(f"Observations: {stats_dict['count']}")
        print(f"Mean: {stats_dict['mean']:.6f}")
        print(f"Std Dev: {stats_dict['std']:.6f}")
        print(f"Min: {stats_dict['min']:.6f}")
        print(f"Max: {stats_dict['max']:.6f}")
        print(f"Skewness: {stats_dict['skewness']:.4f}")
        print(f"Kurtosis: {stats_dict['kurtosis']:.4f}")
        print(f"Jarque-Bera: {stats_dict['jarque_bera'][0]:.4f} (p={stats_dict['jarque_bera'][1]:.4f})")
        print(f"{'='*60}\n")
        
        return stats_dict
    
    def run_full_pipeline(self, return_method: str = 'log') -> pd.Series:
        """
        Execute complete preprocessing pipeline
        
        Parameters:
        -----------
        return_method : str
            Method for computing returns
        
        Returns:
        --------
        pd.Series : Preprocessed returns series
        """
        print(f"\n{'#'*60}")
        print(f"PREPROCESSING PIPELINE: {self.name}")
        print(f"{'#'*60}\n")
        
        # Step 1: Compute returns
        self.compute_returns(method=return_method)
        
        # Step 2: Descriptive statistics
        self.compute_descriptive_stats()
        
        # Step 3: Outlier detection
        self.detect_outliers(method='zscore', threshold=3.0)
        
        # Step 4: Stationarity tests
        stationarity_results = self.check_stationarity()
        
        # Step 5: Apply differencing if needed
        if not stationarity_results['consensus_stationary']:
            print("⚠ Series is non-stationary. Applying differencing...")
            self.apply_differencing(order=1)
            self.check_stationarity(series=self.stationary_series)
        else:
            self.stationary_series = self.returns
        
        print(f"\n{'#'*60}")
        print(f"PREPROCESSING COMPLETE")
        print(f"{'#'*60}\n")
        
        return self.stationary_series


# Example usage and testing
if __name__ == "__main__":
    print("Volatility Forecasting Engine - Data Preprocessing Module")
    print("="*60)
    
    # Generate synthetic financial data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    
    # Simulate price series with volatility clustering
    returns = np.random.normal(0.0005, 0.02, len(dates))
    volatility = np.ones(len(dates)) * 0.02
    
    # Add GARCH-like volatility clustering
    for i in range(1, len(dates)):
        volatility[i] = 0.01 + 0.1 * returns[i-1]**2 + 0.85 * volatility[i-1]
        returns[i] = np.random.normal(0.0005, volatility[i])
    
    prices = 100 * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=dates, name='Synthetic Asset')
    
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor(price_series, name="Synthetic Asset")
    clean_returns = preprocessor.run_full_pipeline(return_method='log')
    
    print(f"\nPreprocessed returns shape: {clean_returns.shape}")
    print(f"Ready for volatility modeling!")
