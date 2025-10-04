# Volatility Forecasting Engine

A comprehensive quantitative finance tool for modeling and forecasting financial volatility using advanced econometric techniques.

## Features

### Core MVP Features
- **Data Preprocessing & Stationarity Checks**: Complete pipeline for cleaning financial time series data, computing log returns, and performing ADF/KPSS tests
- **Volatility Models**: GARCH(1,1) implementation with maximum likelihood estimation
- **Forecasting Framework**: Rolling window forecasts with confidence intervals
- **Evaluation Metrics**: MAPE, MAE, RMSE, QLIKE, Mincer-Zarnowitz tests, and comprehensive diagnostics

### Stretch Features
- **Kalman Filter**: State-space model for latent volatility estimation with MLE parameter estimation
- **Monte Carlo Stress Testing**: Simulation-based risk analysis with fat-tailed distributions (Student-t, skewed-t)
- **Hyperparameter Optimization**: Grid search and Bayesian optimization for model selection
- **Advanced Visualizations**: Professional plots for volatility analysis and forecast evaluation

## Installation

```bash
# Install dependencies
pip install -r scripts/requirements.txt
```

## Quick Start

### 1. Data Preprocessing

```python
from scripts.data_preprocessing import DataPreprocessor
import pandas as pd

# Load your price data
prices = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)

# Run preprocessing pipeline
preprocessor = DataPreprocessor(prices['close'], name="Your Asset")
clean_returns = preprocessor.run_full_pipeline(return_method='log')
```

### 2. GARCH Modeling

```python
from scripts.garch_model import GARCHModel

# Fit GARCH(1,1) model
garch = GARCHModel(clean_returns, p=1, q=1)
garch.fit(dist='normal')

# Run diagnostics
diagnostics = garch.diagnose()

# Generate forecasts
forecasts = garch.forecast(horizon=10)
```

### 3. Rolling Window Forecasting

```python
from scripts.forecasting import VolatilityForecaster

# Fit and forecast
forecaster = VolatilityForecaster(clean_returns)
model, forecast_results = forecaster.fit_and_forecast(
    train_size=0.8,
    horizon=1,
    p=1,
    q=1
)
```

### 4. Evaluation

```python
from scripts.evaluation import ForecastEvaluator, VolatilityVisualizer

# Evaluate forecasts
evaluator = ForecastEvaluator(
    forecasts=forecast_results['forecast'],
    actuals=forecast_results['actual']
)
metrics = evaluator.calculate_all_metrics()

# Visualize results
visualizer = VolatilityVisualizer()
visualizer.plot_forecast_vs_actual(forecast_results)
visualizer.plot_forecast_errors(forecast_results)
```

### 5. Advanced Features

#### Kalman Filter

```python
from scripts.advanced_models import KalmanVolatilityFilter

kalman = KalmanVolatilityFilter(clean_returns)
estimated_params = kalman.estimate_parameters()
kalman_vol = kalman.get_volatility_estimates()
```

#### Monte Carlo Stress Testing

```python
from scripts.advanced_models import MonteCarloStressTester

stress_tester = MonteCarloStressTester(garch, n_simulations=10000)
simulated_paths = stress_tester.simulate_paths(
    horizon=20,
    distribution='t',
    df=5
)
risk_metrics = stress_tester.calculate_risk_metrics()
```

#### Hyperparameter Optimization

```python
from scripts.hyperparameter_optimization import GARCHOptimizer

optimizer = GARCHOptimizer(clean_returns, train_size=0.8)
grid_results = optimizer.grid_search(
    p_range=[1, 2, 3],
    q_range=[1, 2, 3],
    dist_options=['normal', 't'],
    criterion='aic'
)
```

## Project Structure

```
├── scripts/
│   ├── data_preprocessing.py           # Data cleaning and stationarity tests
│   ├── garch_model.py                  # GARCH volatility modeling
│   ├── forecasting.py                  # Rolling window forecasts
│   ├── evaluation.py                   # Performance metrics and visualization
│   ├── advanced_models.py              # Kalman filter and Monte Carlo
│   ├── hyperparameter_optimization.py  # Model selection
│   └── requirements.txt                # Python dependencies
├── README.md
```

## Methodology

### GARCH(1,1) Model

The GARCH(1,1) model captures volatility clustering in financial returns:

$$\sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

where:
- $$\sigma^2_t$$ is the conditional variance at time t
- $$\omega$$ is the constant term
- $$\alpha$$ is the ARCH parameter (reaction to shocks)
- $$\beta$$ is the GARCH parameter (persistence)
- Stationarity requires: $$\alpha + \beta < 1$$

### Kalman Filter State-Space Model

Observation equation: $$r_t = \sigma_t \epsilon_t$$

State equation: $$\log(\sigma^2_t) = \mu + \phi \log(\sigma^2_{t-1}) + \eta_t$$

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error
- **QLIKE**: Quasi-likelihood loss function (robust for volatility)
- **Mincer-Zarnowitz**: Tests forecast unbiasedness
- **Diebold-Mariano**: Compares forecast accuracy between models

### Risk Metrics

- **VaR**: Value at Risk at 95% and 99% confidence levels
- **CVaR**: Conditional Value at Risk (Expected Shortfall)
- **Stress Testing**: Extreme scenario analysis with fat-tailed distributions

## Running Scripts

Execute any module directly:

```bash
# Test preprocessing
python scripts/data_preprocessing.py

# Test GARCH model
python scripts/garch_model.py

# Test forecasting
python scripts/forecasting.py

# Test evaluation
python scripts/evaluation.py

# Test advanced models
python scripts/advanced_models.py

# Test optimization
python scripts/hyperparameter_optimization.py
```

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- SciPy >= 1.10.0
- Statsmodels >= 0.14.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- arch >= 6.2.0

## References

- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
- Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity"
- Diebold, F. X., & Mariano, R. S. (1995). "Comparing Predictive Accuracy"
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"

## License

MIT License
