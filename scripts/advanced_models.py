"""
Advanced Volatility Models - Stretch Features
Kalman filter state-space models and Monte Carlo stress testing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class KalmanVolatilityFilter:
    """
    Kalman filter for latent volatility estimation
    
    State-space representation:
    Observation equation: r_t = σ_t * ε_t
    State equation: log(σ²_t) = μ + φ * log(σ²_{t-1}) + η_t
    
    where ε_t ~ N(0,1) and η_t ~ N(0, σ²_η)
    """
    
    def __init__(self, returns: pd.Series):
        """
        Initialize Kalman filter
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        """
        self.returns = returns.dropna()
        self.n = len(self.returns)
        
        # State variables
        self.filtered_states = None
        self.filtered_covariances = None
        self.smoothed_states = None
        self.smoothed_covariances = None
        
        # Parameters
        self.params = None
        
    def initialize_parameters(self) -> Dict:
        """
        Initialize filter parameters using method of moments
        
        Returns:
        --------
        dict : Initial parameters
        """
        # Use sample variance as initial estimate
        sample_var = self.returns.var()
        
        params = {
            'mu': np.log(sample_var),  # Mean of log-variance
            'phi': 0.95,  # Persistence parameter
            'sigma_eta': 0.1,  # State noise std
            'initial_state': np.log(sample_var),
            'initial_variance': 0.1
        }
        
        return params
    
    def filter(self, params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter (forward pass)
        
        Parameters:
        -----------
        params : dict
            Model parameters
        
        Returns:
        --------
        tuple : (filtered_states, filtered_covariances)
        """
        if params is None:
            params = self.initialize_parameters()
        
        self.params = params
        
        # Extract parameters
        mu = params['mu']
        phi = params['phi']
        sigma_eta = params['sigma_eta']
        
        # Initialize
        filtered_states = np.zeros(self.n)
        filtered_covariances = np.zeros(self.n)
        
        filtered_states[0] = params['initial_state']
        filtered_covariances[0] = params['initial_variance']
        
        # Kalman filter recursion
        for t in range(1, self.n):
            # Prediction step
            state_pred = mu + phi * (filtered_states[t-1] - mu)
            cov_pred = phi**2 * filtered_covariances[t-1] + sigma_eta**2
            
            # Observation variance
            obs_var = np.exp(state_pred)
            
            # Innovation
            innovation = self.returns.iloc[t]**2 - obs_var
            
            # Innovation variance (linearization)
            H = 2 * obs_var  # Derivative of observation equation
            innovation_var = H**2 * cov_pred + obs_var
            
            # Kalman gain
            K = (H * cov_pred) / innovation_var
            
            # Update step
            filtered_states[t] = state_pred + K * innovation
            filtered_covariances[t] = (1 - K * H) * cov_pred
        
        self.filtered_states = filtered_states
        self.filtered_covariances = filtered_covariances
        
        return filtered_states, filtered_covariances
    
    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman smoother (backward pass)
        
        Returns:
        --------
        tuple : (smoothed_states, smoothed_covariances)
        """
        if self.filtered_states is None:
            raise ValueError("Must run filter before smoothing")
        
        phi = self.params['phi']
        mu = self.params['mu']
        sigma_eta = self.params['sigma_eta']
        
        # Initialize with filtered values
        smoothed_states = self.filtered_states.copy()
        smoothed_covariances = self.filtered_covariances.copy()
        
        # Backward recursion
        for t in range(self.n - 2, -1, -1):
            # Prediction
            state_pred = mu + phi * (smoothed_states[t] - mu)
            cov_pred = phi**2 * smoothed_covariances[t] + sigma_eta**2
            
            # Smoother gain
            J = (phi * smoothed_covariances[t]) / cov_pred
            
            # Smoothed estimates
            smoothed_states[t] = self.filtered_states[t] + J * (smoothed_states[t+1] - state_pred)
            smoothed_covariances[t] = self.filtered_covariances[t] + J**2 * (smoothed_covariances[t+1] - cov_pred)
        
        self.smoothed_states = smoothed_states
        self.smoothed_covariances = smoothed_covariances
        
        return smoothed_states, smoothed_covariances
    
    def estimate_parameters(self, method: str = 'mle') -> Dict:
        """
        Estimate parameters via maximum likelihood
        
        Parameters:
        -----------
        method : str
            Estimation method ('mle')
        
        Returns:
        --------
        dict : Estimated parameters
        """
        print(f"\n{'='*60}")
        print(f"KALMAN FILTER PARAMETER ESTIMATION")
        print(f"{'='*60}")
        
        def neg_log_likelihood(theta):
            """Negative log-likelihood for optimization"""
            mu, phi, log_sigma_eta = theta
            
            # Ensure stationarity
            if abs(phi) >= 1:
                return 1e10
            
            params = {
                'mu': mu,
                'phi': phi,
                'sigma_eta': np.exp(log_sigma_eta),
                'initial_state': mu,
                'initial_variance': (np.exp(log_sigma_eta)**2) / (1 - phi**2)
            }
            
            try:
                self.filter(params)
                
                # Calculate log-likelihood
                log_lik = 0
                for t in range(1, self.n):
                    obs_var = np.exp(self.filtered_states[t])
                    log_lik += -0.5 * (np.log(2 * np.pi * obs_var) + 
                                      (self.returns.iloc[t]**2) / obs_var)
                
                return -log_lik
            except:
                return 1e10
        
        # Initial guess
        initial_params = self.initialize_parameters()
        x0 = [initial_params['mu'], initial_params['phi'], 
              np.log(initial_params['sigma_eta'])]
        
        # Optimize
        result = minimize(neg_log_likelihood, x0, method='Nelder-Mead',
                         options={'maxiter': 1000})
        
        # Extract estimated parameters
        mu_est, phi_est, log_sigma_eta_est = result.x
        
        estimated_params = {
            'mu': mu_est,
            'phi': phi_est,
            'sigma_eta': np.exp(log_sigma_eta_est),
            'initial_state': mu_est,
            'initial_variance': (np.exp(log_sigma_eta_est)**2) / (1 - phi_est**2),
            'log_likelihood': -result.fun
        }
        
        print(f"\nEstimated Parameters:")
        print(f"  μ (mean):        {estimated_params['mu']:.6f}")
        print(f"  φ (persistence): {estimated_params['phi']:.6f}")
        print(f"  σ_η (state std): {estimated_params['sigma_eta']:.6f}")
        print(f"  Log-Likelihood:  {estimated_params['log_likelihood']:.2f}")
        print(f"{'='*60}\n")
        
        # Run filter and smoother with estimated parameters
        self.filter(estimated_params)
        self.smooth()
        
        return estimated_params
    
    def get_volatility_estimates(self) -> pd.Series:
        """
        Get volatility estimates from smoothed states
        
        Returns:
        --------
        pd.Series : Volatility estimates
        """
        if self.smoothed_states is None:
            raise ValueError("Must run filter and smoother first")
        
        volatility = np.sqrt(np.exp(self.smoothed_states))
        return pd.Series(volatility, index=self.returns.index, name='Kalman Volatility')


class MonteCarloStressTester:
    """
    Monte Carlo stress testing with fat-tailed distributions
    """
    
    def __init__(self, model, n_simulations: int = 10000):
        """
        Initialize stress tester
        
        Parameters:
        -----------
        model : GARCHModel
            Fitted GARCH model
        n_simulations : int
            Number of Monte Carlo simulations
        """
        self.model = model
        self.n_simulations = n_simulations
        self.simulation_results = None
        
    def simulate_paths(self, horizon: int = 20, 
                      distribution: str = 't',
                      df: float = 5.0) -> np.ndarray:
        """
        Simulate return paths using fat-tailed distributions
        
        Parameters:
        -----------
        horizon : int
            Simulation horizon
        distribution : str
            Distribution for innovations ('normal', 't', 'skewt')
        df : float
            Degrees of freedom for Student-t distribution
        
        Returns:
        --------
        np.ndarray : Simulated paths (n_simulations x horizon)
        """
        print(f"\n{'='*60}")
        print(f"MONTE CARLO STRESS TESTING")
        print(f"{'='*60}")
        print(f"Simulations: {self.n_simulations}")
        print(f"Horizon: {horizon}")
        print(f"Distribution: {distribution}")
        if distribution == 't':
            print(f"Degrees of freedom: {df}")
        print(f"{'='*60}\n")
        
        # Extract GARCH parameters
        omega = self.model.params['omega']
        alpha = self.model.params['alpha']
        beta = self.model.params['beta']
        
        # Get last conditional variance
        last_variance = (self.model.conditional_volatility.iloc[-1] * 100) ** 2
        last_return = self.model.returns.iloc[-1] * 100
        
        # Initialize arrays
        returns = np.zeros((self.n_simulations, horizon))
        variances = np.zeros((self.n_simulations, horizon))
        
        for sim in range(self.n_simulations):
            var_t = last_variance
            ret_t = last_return
            
            for t in range(horizon):
                # Update variance
                var_t = omega + alpha * ret_t**2 + beta * var_t
                variances[sim, t] = var_t
                
                # Generate innovation
                if distribution == 'normal':
                    epsilon = np.random.normal(0, 1)
                elif distribution == 't':
                    epsilon = np.random.standard_t(df)
                elif distribution == 'skewt':
                    # Simplified skewed-t (using mixture)
                    if np.random.rand() < 0.5:
                        epsilon = np.random.standard_t(df)
                    else:
                        epsilon = -np.random.standard_t(df) * 0.8
                else:
                    raise ValueError("distribution must be 'normal', 't', or 'skewt'")
                
                # Generate return
                ret_t = np.sqrt(var_t) * epsilon
                returns[sim, t] = ret_t
            
            # Progress indicator
            if (sim + 1) % 1000 == 0:
                print(f"Progress: {((sim + 1) / self.n_simulations) * 100:.1f}%")
        
        self.simulation_results = {
            'returns': returns,
            'variances': variances,
            'volatilities': np.sqrt(variances)
        }
        
        print(f"\n✓ Simulation complete!")
        
        return returns
    
    def calculate_risk_metrics(self, confidence_levels: list = [0.95, 0.99]) -> Dict:
        """
        Calculate risk metrics from simulations
        
        Parameters:
        -----------
        confidence_levels : list
            Confidence levels for VaR and CVaR
        
        Returns:
        --------
        dict : Risk metrics
        """
        if self.simulation_results is None:
            raise ValueError("Must run simulations first")
        
        returns = self.simulation_results['returns']
        
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(returns, axis=1)
        final_returns = cumulative_returns[:, -1]
        
        print(f"\n{'='*60}")
        print(f"RISK METRICS")
        print(f"{'='*60}")
        
        metrics = {}
        
        for cl in confidence_levels:
            alpha = 1 - cl
            
            # Value at Risk (VaR)
            var = np.percentile(final_returns, alpha * 100)
            
            # Conditional Value at Risk (CVaR / Expected Shortfall)
            cvar = final_returns[final_returns <= var].mean()
            
            metrics[f'VaR_{int(cl*100)}'] = var / 100  # Unscale
            metrics[f'CVaR_{int(cl*100)}'] = cvar / 100
            
            print(f"\nConfidence Level: {cl*100}%")
            print(f"  VaR:  {var/100:.6f}")
            print(f"  CVaR: {cvar/100:.6f}")
        
        # Additional metrics
        metrics['mean_return'] = final_returns.mean() / 100
        metrics['std_return'] = final_returns.std() / 100
        metrics['skewness'] = stats.skew(final_returns)
        metrics['kurtosis'] = stats.kurtosis(final_returns)
        
        print(f"\nDistribution Statistics:")
        print(f"  Mean:     {metrics['mean_return']:.6f}")
        print(f"  Std Dev:  {metrics['std_return']:.6f}")
        print(f"  Skewness: {metrics['skewness']:.4f}")
        print(f"  Kurtosis: {metrics['kurtosis']:.4f}")
        
        print(f"{'='*60}\n")
        
        return metrics
    
    def stress_scenario(self, shock_size: float = 3.0) -> Dict:
        """
        Analyze extreme stress scenario
        
        Parameters:
        -----------
        shock_size : float
            Size of shock in standard deviations
        
        Returns:
        --------
        dict : Stress scenario results
        """
        if self.simulation_results is None:
            raise ValueError("Must run simulations first")
        
        returns = self.simulation_results['returns']
        
        # Find paths with extreme shocks
        min_returns = returns.min(axis=1)
        extreme_paths = returns[min_returns <= -shock_size * returns.std()]
        
        print(f"\n{'='*60}")
        print(f"STRESS SCENARIO ANALYSIS")
        print(f"{'='*60}")
        print(f"Shock size: {shock_size}σ")
        print(f"Extreme paths: {len(extreme_paths)} ({len(extreme_paths)/len(returns)*100:.2f}%)")
        
        if len(extreme_paths) > 0:
            results = {
                'n_extreme_paths': len(extreme_paths),
                'extreme_path_pct': len(extreme_paths) / len(returns) * 100,
                'worst_return': extreme_paths.sum(axis=1).min() / 100,
                'avg_extreme_return': extreme_paths.sum(axis=1).mean() / 100
            }
            
            print(f"Worst cumulative return: {results['worst_return']:.6f}")
            print(f"Average extreme return: {results['avg_extreme_return']:.6f}")
        else:
            results = {'message': 'No extreme paths found'}
            print("No extreme paths found at this threshold")
        
        print(f"{'='*60}\n")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("Volatility Forecasting Engine - Advanced Models Module")
    print("="*60)
    
    # Generate synthetic data
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
    
    returns_series = pd.Series(returns, name='Returns')
    
    # Test 1: Kalman Filter
    print("\n" + "="*60)
    print("TEST 1: KALMAN FILTER")
    print("="*60)
    
    kalman = KalmanVolatilityFilter(returns_series)
    estimated_params = kalman.estimate_parameters()
    kalman_vol = kalman.get_volatility_estimates()
    
    print(f"\nKalman volatility estimates (first 10):")
    print(kalman_vol.head(10))
    
    # Test 2: Monte Carlo Stress Testing
    print("\n" + "="*60)
    print("TEST 2: MONTE CARLO STRESS TESTING")
    print("="*60)
    
    # First fit a GARCH model
    from scripts.garch_model import GARCHModel
    
    garch = GARCHModel(returns_series, p=1, q=1)
    garch.fit(show_summary=False)
    
    # Run stress tests
    stress_tester = MonteCarloStressTester(garch, n_simulations=5000)
    simulated_paths = stress_tester.simulate_paths(horizon=20, distribution='t', df=5)
    
    # Calculate risk metrics
    risk_metrics = stress_tester.calculate_risk_metrics(confidence_levels=[0.95, 0.99])
    
    # Stress scenario
    stress_results = stress_tester.stress_scenario(shock_size=3.0)
    
    print("\n✓ Advanced models testing complete!")
