"""
Bayesian SWR - Real 60/40 VTI/BND Data (2008-2026)
====================================================

Uses actual historical returns for a simple Bogleheads portfolio:
- 60% VTI (Vanguard Total Stock Market, started 2001)
- 40% BND (Vanguard Total Bond Market, started 2007)

This gives 19 years of real data including both the Global Financial Crisis (GFC, 2008) and the subsequent bull market.
"""

import numpy as np
import pymc as pm
import arviz as az

def fetch_and_calculate():
    """Fetch VTI/BND data and calculate portfolio returns."""
    try:
        import yfinance as yf
        # Configure retries for reliability
        yf.config.network.retries = 3
    except ImportError:
        print("Install yfinance: uv pip install yfinance")
        return None
    
    print("Fetching VTI and BND historical data...")
    
    # Fetch data
    vti = yf.Ticker('VTI').history(period='max')['Close']
    bnd = yf.Ticker('BND').history(period='max')['Close']
    
    # Align to common period (both started by 2008)
    common_start = max(vti.index[0], bnd.index[0])
    vti = vti[vti.index >= common_start]
    bnd = bnd[bnd.index >= common_start]
    
    # Annual returns
    vti_ret = vti.resample('YE').last().pct_change().dropna()
    bnd_ret = bnd.resample('YE').last().pct_change().dropna()
    
    # Align years
    common_years = vti_ret.index.intersection(bnd_ret.index)
    vti_ret = vti_ret.loc[common_years]
    bnd_ret = bnd_ret.loc[common_years]
    
    # 60/40 portfolio
    portfolio_returns = 0.6 * vti_ret + 0.4 * bnd_ret
    
    return portfolio_returns.values, portfolio_returns.index.year

def classical_mc(mean, std, years=30, initial=1_000_000, withdrawal=40_000, n_paths=10000):
    """Classical Monte Carlo."""
    np.random.seed(42)
    terminal_wealth = []
    depletion_count = 0
    
    for _ in range(n_paths):
        portfolio = initial
        depleted = False
        for _ in range(years):
            ret = np.random.normal(mean, std)
            portfolio = portfolio * (1 + ret) - withdrawal
            if portfolio <= 0 and not depleted:
                depletion_count += 1
                depleted = True
        terminal_wealth.append(portfolio)
    
    return {
        'success_rate': 1 - (depletion_count / n_paths),
        'median': np.median(terminal_wealth),
        'p5': np.percentile(terminal_wealth, 5),
        'p95': np.percentile(terminal_wealth, 95),
    }

def bayesian_mcmc(historical_returns, n_samples=2000, years=30, initial=1_000_000, withdrawal=40_000):
    """Bayesian MCMC with parameter uncertainty."""
    
    print("Running Bayesian inference...")
    
    with pm.Model() as model:
        mu_prior_mean = np.mean(historical_returns)
        mu_prior_std = np.std(historical_returns) / np.sqrt(len(historical_returns))
        
        mu = pm.Normal('expected_return', mu=mu_prior_mean, sigma=mu_prior_std * 3)
        sigma = pm.HalfNormal('volatility', sigma=np.std(historical_returns))
        
        returns_obs = pm.Normal('historical_returns', mu=mu, sigma=sigma, 
                                observed=historical_returns)
        
        trace = pm.sample(n_samples, tune=1000, cores=4, random_seed=42,
                          return_inferencedata=True)
    
    print(f"Inference complete. R-hat max: {max(az.rhat(trace).values()).values:.3f}")
    
    # Posterior predictive
    print("Running posterior predictive simulation...")
    mu_samples = trace.posterior['expected_return'].values.flatten()
    sigma_samples = trace.posterior['volatility'].values.flatten()
    
    terminal_wealth = []
    depletion_count = 0
    total_paths = 0
    
    for mu_s, sigma_s in zip(mu_samples[:500], sigma_samples[:500]):
        for _ in range(100):
            portfolio = initial
            depleted = False
            for _ in range(years):
                ret = np.random.normal(mu_s, sigma_s)
                portfolio = portfolio * (1 + ret) - withdrawal
                if portfolio <= 0 and not depleted:
                    depletion_count += 1
                    depleted = True
            terminal_wealth.append(portfolio)
            total_paths += 1
    
    return {
        'success_rate': 1 - (depletion_count / total_paths),
        'median': np.median(terminal_wealth),
        'p5': np.percentile(terminal_wealth, 5),
        'p95': np.percentile(terminal_wealth, 95),
        'posterior_mu_mean': np.mean(mu_samples),
        'posterior_mu_std': np.std(mu_samples),
    }

if __name__ == "__main__":
    # Fetch data
    returns, years = fetch_and_calculate()
    
    if returns is None:
        exit(1)
    
    n_years = len(returns)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    sem = std / np.sqrt(n_years)
    
    print(f"\n{'='*60}")
    print(f"60/40 VTI/BND Portfolio - Real Historical Data")
    print(f"{'='*60}")
    print(f"Period: {years[0]}-{years[-1]} ({n_years} years)")
    print(f"\nObserved mean return: {mean*100:.2f}%")
    print(f"Observed volatility: {std*100:.2f}%")
    print(f"Standard Error of Mean: {sem*100:.2f}%")
    print(f"95% CI for true mean: [{mean - 1.96*sem:.3f}, {mean + 1.96*sem:.3f}]")
    
    print(f"\n{'-'*60}")
    print("Classical Monte Carlo")
    print(f"{'-'*60}")
    classical = classical_mc(mean, std)
    print(f"Success rate: {classical['success_rate']*100:.1f}%")
    print(f"Median terminal wealth: ${classical['median']:,.0f}")
    print(f"5th percentile: ${classical['p5']:,.0f}")
    
    print(f"\n{'-'*60}")
    print("Bayesian MCMC")
    print(f"{'-'*60}")
    bayesian = bayesian_mcmc(returns)
    print(f"Success rate: {bayesian['success_rate']*100:.1f}%")
    print(f"Median terminal wealth: ${bayesian['median']:,.0f}")
    print(f"5th percentile: ${bayesian['p5']:,.0f}")
    print(f"Posterior μ: {bayesian['posterior_mu_mean']*100:.2f}% ± {bayesian['posterior_mu_std']*100:.2f}%")
    
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    diff = (bayesian['success_rate'] - classical['success_rate']) * 100
    print(f"Success rate difference: {diff:+.1f} percentage points")
    print(f"  Classical MC: {classical['success_rate']*100:.1f}%")
    print(f"  Bayesian MCMC: {bayesian['success_rate']*100:.1f}%")
