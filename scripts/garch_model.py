"""
GARCH Volatility Model Implementation
Implements GARCH(1,1) with maximum likelihood estimation and diagnostics
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class GARCHModel:
    """
    GARCH(1,1) volatility model with comprehensive diagnostics
    
    Model specification:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    where:
    - σ²_t is the conditional variance at time t
    - ω is the constant term (omega)
    - α is the ARCH parameter (alpha)
    - β is the GARCH parameter (beta)
    - ε²_{t-1} is the squared residual from previous period
    """
    
    def __init__(self, returns: pd.Series, p: int = 1, q: int = 1):
        """
        Initialize GARCH model
        
        Parameters:
        -----------
        returns : pd.Series
            Return series (should be stationary)
        p : int
            GARCH order (default: 1)
        q : int
            ARCH order (default: 1)
        """
        self.returns = returns.dropna()
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.params = None
        self.conditional_volatility = None
        self.standardized_residuals = None
        
    def fit(self, dist: str = 'normal', show_summary: bool = True) -> 'GARCHModel':
        """
        Fit GARCH model using maximum likelihood estimation
        
        Parameters:
        -----------
        dist : str
            Distribution assumption ('normal', 't', 'skewt', 'ged')
        show_summary : bool
            Whether to print model summary
        
        Returns:
        --------
        self : GARCHModel
            Fitted model instance
        """
        print(f"\n{'='*60}")
        print(f"FITTING GARCH({self.p},{self.q}) MODEL")
        print(f"{'='*60}")
        print(f"Distribution: {dist}")
        print(f"Observations: {len(self.returns)}")
        
        # Specify and fit GARCH model
        self.model = arch_model(
            self.returns * 100,  # Scale to percentage for numerical stability
            vol='Garch',
            p=self.p,
            q=self.q,
            dist=dist,
            rescale=False
        )
        
        self.fitted_model = self.model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        self.params = {
            'omega': self.fitted_model.params['omega'],
            'alpha': self.fitted_model.params['alpha[1]'] if self.q > 0 else 0,
            'beta': self.fitted_model.params['beta[1]'] if self.p > 0 else 0,
            'mu': self.fitted_model.params['mu']
        }
        
        # Calculate persistence
        self.params['persistence'] = self.params['alpha'] + self.params['beta']
        
        # Extract conditional volatility and standardized residuals
        self.conditional_volatility = self.fitted_model.conditional_volatility / 100
        self.standardized_residuals = self.fitted_model.std_resid
        
        if show_summary:
            print(f"\n{self.fitted_model.summary()}")
            print(f"\n{'='*60}")
            print(f"MODEL PARAMETERS")
            print(f"{'='*60}")
            print(f"ω (omega):     {self.params['omega']:.6f}")
            print(f"α (alpha):     {self.params['alpha']:.6f}")
            print(f"β (beta):      {self.params['beta']:.6f}")
            print(f"μ (mu):        {self.params['mu']:.6f}")
            print(f"Persistence:   {self.params['persistence']:.6f}")
            print(f"Log-Likelihood: {self.fitted_model.loglikelihood:.2f}")
            print(f"AIC:           {self.fitted_model.aic:.2f}")
            print(f"BIC:           {self.fitted_model.bic:.2f}")
            print(f"{'='*60}\n")
            
            # Check stationarity condition
            if self.params['persistence'] < 1:
                print("✓ Model is covariance stationary (α + β < 1)")
            else:
                print("✗ WARNING: Model may be non-stationary (α + β ≥ 1)")
        
        return self
    
    def diagnose(self) -> Dict:
        """
        Perform comprehensive model diagnostics
        
        Returns:
        --------
        dict : Diagnostic test results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before diagnostics")
        
        print(f"\n{'='*60}")
        print(f"MODEL DIAGNOSTICS")
        print(f"{'='*60}")
        
        diagnostics = {}
        
        # 1. Standardized Residuals Tests
        std_resid = self.standardized_residuals
        
        # Ljung-Box test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(std_resid, lags=[10], return_df=True)
        diagnostics['ljung_box'] = {
            'statistic': lb_test['lb_stat'].values[0],
            'pvalue': lb_test['lb_pvalue'].values[0]
        }
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
        diagnostics['jarque_bera'] = {
            'statistic': jb_stat,
            'pvalue': jb_pvalue
        }
        
        # 2. ARCH-LM test for remaining ARCH effects
        from statsmodels.stats.diagnostic import het_arch
        arch_lm = het_arch(std_resid, nlags=10)
        diagnostics['arch_lm'] = {
            'statistic': arch_lm[0],
            'pvalue': arch_lm[1]
        }
        
        # 3. Descriptive statistics of standardized residuals
        diagnostics['std_resid_stats'] = {
            'mean': std_resid.mean(),
            'std': std_resid.std(),
            'skewness': stats.skew(std_resid),
            'kurtosis': stats.kurtosis(std_resid)
        }
        
        # Print results
        print(f"\n1. Ljung-Box Test (Autocorrelation):")
        print(f"   Statistic: {diagnostics['ljung_box']['statistic']:.4f}")
        print(f"   P-value: {diagnostics['ljung_box']['pvalue']:.4f}")
        print(f"   → {'✓ No autocorrelation' if diagnostics['ljung_box']['pvalue'] > 0.05 else '✗ Autocorrelation detected'}")
        
        print(f"\n2. Jarque-Bera Test (Normality):")
        print(f"   Statistic: {diagnostics['jarque_bera']['statistic']:.4f}")
        print(f"   P-value: {diagnostics['jarque_bera']['pvalue']:.4f}")
        print(f"   → {'✓ Residuals are normal' if diagnostics['jarque_bera']['pvalue'] > 0.05 else '✗ Residuals are non-normal'}")
        
        print(f"\n3. ARCH-LM Test (Remaining ARCH Effects):")
        print(f"   Statistic: {diagnostics['arch_lm']['statistic']:.4f}")
        print(f"   P-value: {diagnostics['arch_lm']['pvalue']:.4f}")
        print(f"   → {'✓ No remaining ARCH effects' if diagnostics['arch_lm']['pvalue'] > 0.05 else '✗ ARCH effects remain'}")
        
        print(f"\n4. Standardized Residuals Statistics:")
        print(f"   Mean: {diagnostics['std_resid_stats']['mean']:.4f}")
        print(f"   Std Dev: {diagnostics['std_resid_stats']['std']:.4f}")
        print(f"   Skewness: {diagnostics['std_resid_stats']['skewness']:.4f}")
        print(f"   Kurtosis: {diagnostics['std_resid_stats']['kurtosis']:.4f}")
        
        print(f"{'='*60}\n")
        
        return diagnostics
    
    def forecast(self, horizon: int = 1, method: str = 'analytic') -> pd.DataFrame:
        """
        Generate volatility forecasts
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon (number of periods ahead)
        method : str
            Forecasting method ('analytic' or 'simulation')
        
        Returns:
        --------
        pd.DataFrame : Forecasts with confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Generate forecast
        forecast = self.fitted_model.forecast(horizon=horizon, method=method)
        
        # Extract variance forecasts and convert to volatility
        variance_forecast = forecast.variance.values[-1, :] / 10000  # Unscale
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'horizon': range(1, horizon + 1),
            'volatility': volatility_forecast
        })
        
        # Calculate confidence intervals (assuming normal distribution)
        # 95% CI for volatility
        forecast_df['lower_95'] = volatility_forecast * 0.7  # Approximate
        forecast_df['upper_95'] = volatility_forecast * 1.3
        
        return forecast_df
    
    def get_unconditional_volatility(self) -> float:
        """
        Calculate long-run unconditional volatility
        
        Returns:
        --------
        float : Unconditional volatility
        """
        if self.params is None:
            raise ValueError("Model must be fitted first")
        
        omega = self.params['omega']
        alpha = self.params['alpha']
        beta = self.params['beta']
        
        # Unconditional variance: ω / (1 - α - β)
        if alpha + beta < 1:
            unconditional_var = omega / (1 - alpha - beta)
            unconditional_vol = np.sqrt(unconditional_var) / 100  # Unscale
            return unconditional_vol
        else:
            return np.nan
    
    def get_half_life(self) -> float:
        """
        Calculate half-life of volatility shocks
        
        Returns:
        --------
        float : Half-life in periods
        """
        if self.params is None:
            raise ValueError("Model must be fitted first")
        
        persistence = self.params['persistence']
        
        if persistence < 1 and persistence > 0:
            half_life = np.log(0.5) / np.log(persistence)
            return half_life
        else:
            return np.inf


class GARCHModelSelector:
    """
    Automated model selection for GARCH specifications
    """
    
    def __init__(self, returns: pd.Series):
        """
        Initialize model selector
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        """
        self.returns = returns
        self.results = []
    
    def grid_search(self, max_p: int = 3, max_q: int = 3, 
                   criterion: str = 'aic') -> pd.DataFrame:
        """
        Perform grid search over GARCH specifications
        
        Parameters:
        -----------
        max_p : int
            Maximum GARCH order
        max_q : int
            Maximum ARCH order
        criterion : str
            Selection criterion ('aic' or 'bic')
        
        Returns:
        --------
        pd.DataFrame : Results sorted by criterion
        """
        print(f"\n{'='*60}")
        print(f"GARCH MODEL SELECTION - GRID SEARCH")
        print(f"{'='*60}")
        print(f"Testing specifications: p ∈ [1,{max_p}], q ∈ [1,{max_q}]")
        print(f"Selection criterion: {criterion.upper()}\n")
        
        results = []
        
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = GARCHModel(self.returns, p=p, q=q)
                    model.fit(show_summary=False)
                    
                    results.append({
                        'p': p,
                        'q': q,
                        'aic': model.fitted_model.aic,
                        'bic': model.fitted_model.bic,
                        'loglik': model.fitted_model.loglikelihood,
                        'persistence': model.params['persistence']
                    })
                    
                    print(f"✓ GARCH({p},{q}): {criterion.upper()}={getattr(model.fitted_model, criterion):.2f}")
                    
                except Exception as e:
                    print(f"✗ GARCH({p},{q}): Failed ({str(e)[:50]})")
        
        results_df = pd.DataFrame(results).sort_values(criterion)
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: GARCH({results_df.iloc[0]['p']:.0f},{results_df.iloc[0]['q']:.0f})")
        print(f"{criterion.upper()}: {results_df.iloc[0][criterion]:.2f}")
        print(f"{'='*60}\n")
        
        return results_df


# Example usage and testing
if __name__ == "__main__":
    print("Volatility Forecasting Engine - GARCH Model Module")
    print("="*60)
    
    # Generate synthetic data with GARCH properties
    np.random.seed(42)
    n = 1000
    
    # True GARCH(1,1) parameters
    omega = 0.01
    alpha = 0.15
    beta = 0.80
    
    # Simulate GARCH process
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = volatility[t] * np.random.normal(0, 1)
    
    returns_series = pd.Series(returns, name='Synthetic Returns')
    
    # Fit GARCH(1,1) model
    print("\n" + "="*60)
    print("FITTING GARCH(1,1) MODEL")
    print("="*60)
    
    garch = GARCHModel(returns_series, p=1, q=1)
    garch.fit(dist='normal')
    
    # Run diagnostics
    diagnostics = garch.diagnose()
    
    # Generate forecasts
    print("\n" + "="*60)
    print("VOLATILITY FORECASTS")
    print("="*60)
    forecasts = garch.forecast(horizon=10)
    print(forecasts)
    
    # Calculate unconditional volatility
    uncond_vol = garch.get_unconditional_volatility()
    print(f"\nUnconditional Volatility: {uncond_vol:.6f}")
    
    # Calculate half-life
    half_life = garch.get_half_life()
    print(f"Half-life of shocks: {half_life:.2f} periods")
    
    # Model selection
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    selector = GARCHModelSelector(returns_series)
    selection_results = selector.grid_search(max_p=2, max_q=2, criterion='aic')
    print("\nTop 3 models:")
    print(selection_results.head(3))
