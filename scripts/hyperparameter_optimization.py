"""
Hyperparameter Optimization
Grid search and Bayesian optimization for GARCH models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scripts.garch_model import GARCHModel
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class GARCHOptimizer:
    """
    Hyperparameter optimization for GARCH models
    """
    
    def __init__(self, returns: pd.Series, train_size: float = 0.8):
        """
        Initialize optimizer
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        train_size : float
            Proportion for training set
        """
        self.returns = returns
        split_idx = int(len(returns) * train_size)
        self.train_data = returns.iloc[:split_idx]
        self.test_data = returns.iloc[split_idx:]
        
    def grid_search(self, 
                   p_range: List[int] = [1, 2],
                   q_range: List[int] = [1, 2],
                   dist_options: List[str] = ['normal', 't'],
                   criterion: str = 'aic') -> pd.DataFrame:
        """
        Grid search over GARCH specifications
        
        Parameters:
        -----------
        p_range : list
            GARCH orders to test
        q_range : list
            ARCH orders to test
        dist_options : list
            Distributions to test
        criterion : str
            Selection criterion ('aic', 'bic', 'loglik')
        
        Returns:
        --------
        pd.DataFrame : Results sorted by criterion
        """
        print(f"\n{'='*60}")
        print(f"GRID SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        print(f"p range: {p_range}")
        print(f"q range: {q_range}")
        print(f"Distributions: {dist_options}")
        print(f"Criterion: {criterion}")
        print(f"{'='*60}\n")
        
        results = []
        total_combinations = len(p_range) * len(q_range) * len(dist_options)
        current = 0
        
        for p, q, dist in product(p_range, q_range, dist_options):
            current += 1
            try:
                # Fit model
                model = GARCHModel(self.train_data, p=p, q=q)
                model.fit(dist=dist, show_summary=False)
                
                # Calculate out-of-sample forecast error
                from scripts.forecasting import RollingWindowForecaster
                forecaster = RollingWindowForecaster(
                    self.returns,
                    window_type='expanding',
                    window_size=len(self.train_data)
                )
                
                forecast_results = forecaster.forecast_rolling(
                    start_idx=len(self.train_data),
                    end_idx=len(self.train_data) + min(50, len(self.test_data)),
                    horizon=1,
                    refit_frequency=10,
                    p=p,
                    q=q
                )
                
                oos_mae = forecast_results['abs_error'].mean()
                oos_rmse = np.sqrt(forecast_results['squared_error'].mean())
                
                results.append({
                    'p': p,
                    'q': q,
                    'distribution': dist,
                    'aic': model.fitted_model.aic,
                    'bic': model.fitted_model.bic,
                    'loglik': model.fitted_model.loglikelihood,
                    'persistence': model.params['persistence'],
                    'oos_mae': oos_mae,
                    'oos_rmse': oos_rmse
                })
                
                print(f"[{current}/{total_combinations}] GARCH({p},{q})-{dist}: "
                      f"{criterion}={getattr(model.fitted_model, criterion):.2f}, "
                      f"OOS MAE={oos_mae:.6f}")
                
            except Exception as e:
                print(f"[{current}/{total_combinations}] GARCH({p},{q})-{dist}: Failed ({str(e)[:40]})")
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Sort by criterion (lower is better for AIC/BIC, higher for loglik)
            if criterion in ['aic', 'bic', 'oos_mae', 'oos_rmse']:
                results_df = results_df.sort_values(criterion)
            else:
                results_df = results_df.sort_values(criterion, ascending=False)
            
            print(f"\n{'='*60}")
            print(f"BEST MODEL")
            print(f"{'='*60}")
            best = results_df.iloc[0]
            print(f"Specification: GARCH({best['p']:.0f},{best['q']:.0f})-{best['distribution']}")
            print(f"AIC: {best['aic']:.2f}")
            print(f"BIC: {best['bic']:.2f}")
            print(f"Log-Likelihood: {best['loglik']:.2f}")
            print(f"Persistence: {best['persistence']:.4f}")
            print(f"OOS MAE: {best['oos_mae']:.6f}")
            print(f"OOS RMSE: {best['oos_rmse']:.6f}")
            print(f"{'='*60}\n")
        
        return results_df
    
    def bayesian_optimization(self, n_iterations: int = 20) -> Dict:
        """
        Bayesian optimization for continuous hyperparameters
        (Simplified version using random search with adaptive sampling)
        
        Parameters:
        -----------
        n_iterations : int
            Number of optimization iterations
        
        Returns:
        --------
        dict : Best parameters found
        """
        print(f"\n{'='*60}")
        print(f"BAYESIAN OPTIMIZATION (Simplified)")
        print(f"{'='*60}")
        print(f"Iterations: {n_iterations}")
        print(f"{'='*60}\n")
        
        # Parameter space (for demonstration, we'll optimize initial variance)
        best_score = np.inf
        best_params = None
        scores = []
        
        for i in range(n_iterations):
            # Sample parameters (simplified - in practice use GP)
            p = np.random.choice([1, 2])
            q = np.random.choice([1, 2])
            
            try:
                model = GARCHModel(self.train_data, p=p, q=q)
                model.fit(show_summary=False)
                
                score = model.fitted_model.aic
                scores.append(score)
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'p': p,
                        'q': q,
                        'aic': score,
                        'params': model.params
                    }
                
                print(f"Iteration {i+1}/{n_iterations}: GARCH({p},{q}), AIC={score:.2f}")
                
            except Exception as e:
                print(f"Iteration {i+1}/{n_iterations}: Failed")
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Best model: GARCH({best_params['p']},{best_params['q']})")
        print(f"Best AIC: {best_params['aic']:.2f}")
        print(f"{'='*60}\n")
        
        return best_params


# Example usage
if __name__ == "__main__":
    print("Volatility Forecasting Engine - Hyperparameter Optimization")
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
    
    # Test grid search
    optimizer = GARCHOptimizer(returns_series, train_size=0.8)
    grid_results = optimizer.grid_search(
        p_range=[1, 2],
        q_range=[1, 2],
        dist_options=['normal', 't'],
        criterion='aic'
    )
    
    print("\nTop 3 models:")
    print(grid_results.head(3))
    
    # Test Bayesian optimization
    bayesian_results = optimizer.bayesian_optimization(n_iterations=10)
