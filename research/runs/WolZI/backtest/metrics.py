"""Performance metrics including Deflated Sharpe Ratio."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats


def annualized_sharpe(monthly_rets: pd.Series, rf_monthly: float = 0.0) -> float:
    excess = monthly_rets - rf_monthly
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * math.sqrt(12)


def annualized_cagr(monthly_rets: pd.Series) -> float:
    cum = (1 + monthly_rets).prod()
    n_years = len(monthly_rets) / 12.0
    return cum ** (1 / n_years) - 1 if n_years > 0 else np.nan


def max_drawdown(monthly_rets: pd.Series) -> float:
    cum = (1 + monthly_rets).cumprod()
    roll_max = cum.cummax()
    return (cum / roll_max - 1).min()


def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Args:
        sharpe_obs: Observed annualized Sharpe (divided by sqrt(12) for monthly).
        n_obs: Number of monthly observations.
        n_trials: Total number of hyperparameter combinations ever tried.
        skew: Skewness of monthly returns.
        kurt: Excess kurtosis of monthly returns (kurt=3 → normal).
    Returns:
        Probability that the true Sharpe > 0.
    """
    # Convert to monthly Sharpe for the formula
    sr_monthly = sharpe_obs / math.sqrt(12)

    # Expected maximum Sharpe from n_trials iid trials
    # E[max SR] ≈ (1 - euler_gamma) * Z^{-1}(1 - 1/n) + euler_gamma * Z^{-1}(1 - 1/(n*e))
    euler_gamma = 0.5772156649
    if n_trials <= 1:
        sr_max = 0.0
    else:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        sr_max = (1 - euler_gamma) * z1 + euler_gamma * z2

    # SE of SR estimate under non-normality (Bailey & LdP 2014, eq. 4)
    # non_norm is the standard error of the estimated monthly SR
    non_norm = math.sqrt(
        (1 - skew * sr_monthly + (kurt - 1) / 4.0 * sr_monthly ** 2) / (n_obs - 1)
    )

    # sr_max is in "standard normal" units; scale to monthly SR units: divide by sqrt(n_obs)
    # because under null (true SR=0), each monthly SR estimate ~ N(0, 1/sqrt(n_obs))
    sr_max_scaled = sr_max / math.sqrt(n_obs)

    z = (sr_monthly - sr_max_scaled) / non_norm if non_norm > 0 else 0.0
    return float(stats.norm.cdf(z))


def block_bootstrap_sharpe(
    monthly_rets: pd.Series,
    block_len: int = 6,
    n_iter: int = 1000,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Block bootstrap Sharpe ratio distribution."""
    rng = np.random.default_rng(rng_seed)
    n = len(monthly_rets)
    sharpes = []
    arr = monthly_rets.values

    for _ in range(n_iter):
        # Draw blocks with replacement
        blocks = []
        total = 0
        while total < n:
            start = rng.integers(0, n)
            end = min(start + block_len, n)
            blocks.append(arr[start:end])
            total += end - start
        boot = np.concatenate(blocks)[:n]
        s = boot.mean() / boot.std() * math.sqrt(12) if boot.std() > 0 else np.nan
        if not np.isnan(s):
            sharpes.append(s)

    sharpes_arr = np.array(sharpes)
    return {
        "p5": float(np.percentile(sharpes_arr, 5)),
        "p50": float(np.percentile(sharpes_arr, 50)),
        "p95": float(np.percentile(sharpes_arr, 95)),
        "mean": float(sharpes_arr.mean()),
    }
