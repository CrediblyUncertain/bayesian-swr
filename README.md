# Bayesian Safe Withdrawal Rate (SWR) Analysis

A Bayesian MCMC approach to retirement planning that explicitly models parameter uncertainty.

## Overview

This repository demonstrates a Bayesian alternative to classical Monte Carlo simulation for Safe Withdrawal Rate (SWR) analysis. The key insight: classical MC treats estimated parameters as known constants, while Bayesian MCMC properly accounts for parameter uncertainty.

## The Problem: Parameter Uncertainty in Classical Monte Carlo

Classical Monte Carlo simulation for retirement planning typically follows this logic:

1. Estimate mean return (μ̂) and volatility (σ̂) from historical data
2. Treat these estimates as the *true* population parameters
3. Simulate future paths using these fixed values

**The issue:** Even with substantial historical data (e.g., Robert Shiller's 150-year dataset), our estimate of the true mean return has substantial uncertainty. To illustrate: with 150 observations and ~18% annual volatility:

- Standard Error of the Mean = 18%/√150 ≈ **1.47%**
- 95% CI for true mean return: approximately **[4.1%, 9.9%]** (assuming 7% sample mean)

Classical MC assumes we know the answer is exactly 7%. In reality, any value in that range is consistent with the historical data. This is what statisticians call **epistemic uncertainty** (uncertainty in our knowledge) as opposed to **aleatoric uncertainty** (inherent randomness in future returns).

*(Note: The analysis below uses 19 years of actual ETF data, where this uncertainty is even more pronounced.)*

### What About Historical Sequence Analysis?

Some Monte Carlo implementations offer the option to use actual historical return sequences (bootstrapping or block-sampling from observed data) rather than generating random paths from estimated parameters. This approach has intuitive appeal—we're using "real" returns rather than simulated ones.

However, historical sequence analysis faces its own limitations:

- **Small sample:** With 150 years of data, you only get ~120 unique 30-year retirement periods. This is fewer independent samples than typical Monte Carlo simulations generate, potentially leading to noisy percentile estimates.
- **Survivorship bias:** Our historical dataset comes from markets that survived and thrived. We have limited or no data from markets that experienced catastrophic failure (e.g., Russia 1917, China 1949, various emerging market collapses).
- **Structural breaks:** The last 150 years spans the gold standard, Bretton Woods, fiat currency regimes, and multiple inflationary/deflationary periods. Returns from 1871 may not come from the same statistical distribution as returns today.
- **Temporal dependence:** Real returns exhibit momentum, mean-reversion, and volatility clustering that random sampling may miss—but they also reflect specific historical contingencies that may not repeat.

Historical sequence analysis is a valuable complement to parametric Monte Carlo, but it doesn't solve the fundamental problem: we have limited data and substantial uncertainty about whether the future will resemble any particular historical period.

## A Bayesian Alternative

Bayesian MCMC (Markov Chain Monte Carlo) approaches this differently:

1. **Inference:** Use historical data to compute a posterior distribution over plausible values of (μ, σ)
2. **Simulation:** For each plausible parameter set from the posterior, simulate future retirement outcomes
3. **Aggregation:** Combine results across all parameter sets to get the full distribution of outcomes

This properly accounts for both:
- Aleatoric uncertainty: Returns are random even if we knew the true parameters
- Epistemic uncertainty: We don't know the true parameters, only have estimates

**A useful mental model:**
> Classical MC = "We know the dice, now roll them many times."
> Bayesian MCMC = "We're not even sure what dice we're using, so sample both the dice *and* the rolls."

That second layer is exactly what widens the distribution.

## Results: Classical vs. Bayesian

We applied both methods to a classic 60/40 portfolio using real historical data.

**Portfolio:** 60% VTI (Vanguard Total Stock Market), 40% BND (Vanguard Total Bond Market)  
**Data period:** 2008–2026 (19 years of actual ETF returns)  
**Note:** This classic 60/40 allocation—60% equities, 40% bonds—is one of the most widely recommended portfolios, providing broad US market exposure with Treasury stability.  
**Parameters:** 30-year retirement, $1M initial, $40K annual withdrawal (4% SWR)

### Classical Monte Carlo
- **Success rate:** 99.5%
- **Median terminal wealth:** $5.2M
- **5th percentile:** $1.0M
- **Observed mean return:** 8.48%
- **Observed volatility:** 11.50%

### Bayesian MCMC
- **Success rate:** 95.7%
- **Median terminal wealth:** $5.2M  
- **5th percentile:** **$91K** (91% lower)
- **Posterior mean return:** 8.48% ± 2.64%
- **R-hat convergence:** 1.002 (excellent)

**Key finding:** Classical MC shows a 99.5% success rate while Bayesian MCMC shows 95.7%—a 3.8 percentage point difference reflecting parameter uncertainty. More significantly, the 5th percentile terminal wealth drops from $1.0M to $91K, revealing tail risk (the risk of extreme negative outcomes in the worst-case scenarios) that classical methods understate.

## Why This Matters

The success rate difference (99.5% vs. 95.7%) is modest but meaningful. The real story is in the tail risk: **classical MC shows a $1.0M 5th percentile while Bayesian MCMC shows $91K—a 91% difference.**

With only 19 years of ETF history, the Standard Error of the Mean is 2.64% on an 8.48% observed return (31% relative uncertainty). Classical MC ignores this uncertainty; Bayesian MCMC properly incorporates it, revealing that outcomes in the left tail can be substantially worse than point estimates suggest.

For understanding retirement planning uncertainty:

1. **Tail risk awareness:** The 5th percentile isn't just "lower returns"—it can be dramatically lower terminal wealth
2. **Data limitations:** Even nearly two decades of historical data leaves substantial uncertainty in expected return estimates
3. **Methodological humility:** Bayesian credible intervals properly reflect our genuine uncertainty about the future

## Technical Details

**Model specification (classic 60/40 example):**
```
Portfolio: 60% VTI, 40% BND
Data: 19 years actual returns (2008–2026)
Prior: μ ~ Normal(8.48%, 3×SEM), where SEM = 2.64%
       σ ~ Half-Normal(11.50%)
Likelihood: Historical returns ~ Normal(μ, σ²)
Sampler: NUTS (No-U-Turn Sampler)
Convergence: R-hat = 1.002 (excellent)
```

**Prior rationale:** The prior for μ is centered on the observed sample mean (8.48%) with a standard deviation of 3×SEM (7.92%). This is intentionally weakly informative—it allows the posterior to be driven primarily by the data while preventing extreme parameter values that are inconsistent with historical experience. The 3×SEM width ensures we don't encode strong prior beliefs about the true expected return, but we do rule out implausible values (e.g., μ < -5% or μ > 25%) that have no support in 150+ years of financial market history. As the sample size increases, the prior's influence diminishes and the posterior converges to the likelihood.

**Implementation:** PyMC 5.x with ArviZ for diagnostics.

**Limitations:**
- Assumes returns are normally distributed (fat tails not modeled). In reality, moving to a fat-tailed distribution (e.g., Student's t or skew-t) would likely reveal even more extreme left-tail risk for *both* methods.
- Assumes stationarity (no regime switching). A Markov-switching model (e.g., high/low inflation regimes) would likely matter more than parameter uncertainty alone.
- The prior anchors to the sample mean (8.48%). A more skeptical prior (e.g., shrinking toward long-run equity risk premium of 4-6%) would lower success rates further and widen the Bayesian/classical gap.
- The 2008-2026 sample includes a strong post-GFC bull run and misses 1970s stagflation, potentially biasing both mean estimates and posterior dispersion.
- Limited historical data (19 years) for modern ETFs—this is intentionally part of the analysis to highlight parameter uncertainty, but it amplifies the effect.
- Computational cost: ~2 seconds vs. milliseconds for classical MC

**Interpretation nuance:** The $91K 5th percentile shouldn't be taken literally as "your true 5th percentile is $91K." It's better interpreted as: *"If we admit we don't know the true expected return, downside outcomes are much more sensitive than classical MC suggests."* Model specification (fat tails, regimes, priors) likely matters more than the Bayesian vs. classical distinction once you move beyond Gaussian IID assumptions.

**Additional testing:** We also ran the same analysis on synthetic equity data (7% expected return, 18% volatility) with 150 years of history. Classical MC showed 83.9% success vs. 83.7% for Bayesian MCMC—a difference of only 0.2 percentage points (pp). The much smaller gap occurs because longer data reduces parameter uncertainty (SEM = 1.47% vs. 2.64% for the 19-year ETF history).

Notably, even with 150 years of data, the Bayesian 90% credible interval was 67% wider than the classical MC interval ($26.3M vs. $15.8M). This demonstrates that parameter uncertainty affects tail risk quantification regardless of sample size—though the effect is more pronounced with limited historical data.

This confirms that the classic 60/40 results (3.8 pp difference) reflect the higher uncertainty from limited historical data for modern ETFs.

## Installation & Usage

**Requirements:**
- Python 3.10+
- PyMC 5.x
- ArviZ
- NumPy
- Pandas
- yfinance

**To run:**
```bash
uv run --with pymc,arviz,numpy,scipy,yfinance python bayesian_swr.py
```

Or install dependencies manually:
```bash
uv pip install pymc arviz numpy pandas yfinance
python bayesian_swr.py
```

## What the Results Show

**For educational purposes only:**

The classic 60/40 analysis illustrates how parameter uncertainty affects retirement planning analysis:

**Methodological insight:** Classical MC assumes we know the population parameters (μ, σ) with certainty. Bayesian MCMC acknowledges these are estimated from limited samples and incorporates that uncertainty into the analysis.

**Observed difference:** Classical MC shows a 99.5% success rate while Bayesian MCMC shows 95.7%. This 3.8 percentage point difference reflects how parameter uncertainty propagates through 30-year projections.

**Tail risk implications:** The 5th percentile terminal wealth differs substantially ($1.0M vs. $91K), suggesting classical methods may understate the range of possible outcomes in the left tail.

**Data limitations:** With only 19 years of ETF history, the Standard Error of the Mean is 2.64% on an 8.48% observed return — substantial uncertainty even with nearly two decades of data.

**Comparison:** Using synthetic 150-year equity data (which has lower relative uncertainty), the difference shrinks to 0.2 percentage points. This confirms that the classic 60/40 results are driven by the limited historical data available for modern ETFs.

Users can draw their own conclusions about how these methodological differences might inform their understanding of retirement planning uncertainty.

## Connection to Existing Research

This approach aligns with concerns raised by Campbell Harvey and others about data mining and false precision in financial modeling. Just as the "factor zoo" research showed that many claimed factors fail under rigorous statistical scrutiny, classical MC may give false precision in retirement planning by ignoring estimation error.

The solution isn't to abandon Monte Carlo—it's to use it more rigorously by acknowledging that our parameter estimates have uncertainty, even with 150 years of data.

## References

Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.

Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. *Journal of Machine Learning Research*, 15, 1593-1623.

Shiller, R. J. (2022). *Irrational Exuberance* (3rd ed.). Princeton University Press.

---

*Disclaimer: This content is presented for educational and academic purposes only. It describes a statistical methodology and presents comparative results. The code is offered as-is for educational use. Users are solely responsible for their own financial decisions and should consult qualified professionals before acting on any information contained herein.*
