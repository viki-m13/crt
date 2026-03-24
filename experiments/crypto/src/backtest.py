"""
Crypto Backtesting Framework
==============================
Walk-forward backtesting with strict anti-leakage controls.
Adapted for crypto: 365 days/year, higher transaction costs, BTC benchmark.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .strategy import CryptoTMDArcStrategy, CryptoStrategyConfig
from .features import compute_all_features
from .data_pipeline import TRADING_DAYS_PER_YEAR


class BacktestEngine:
    """Walk-forward backtesting engine for crypto."""

    def __init__(self, data_dict, market_ticker="BTC-USD",
                 config: Optional[CryptoStrategyConfig] = None):
        self.data_dict = data_dict
        self.market_ticker = market_ticker
        self.config = config or CryptoStrategyConfig()
        self.strategy = CryptoTMDArcStrategy(self.config)

        print("Pre-computing features...")
        self.features_cache = {}

        btc_close = None
        if market_ticker in data_dict:
            btc_close = data_dict[market_ticker]["Close"]

        for ticker, df in data_dict.items():
            if "Close" not in df.columns:
                continue
            try:
                volume = df.get("Volume")
                # Don't compute cascade for BTC vs itself
                leader = btc_close if ticker != market_ticker else None
                feats = compute_all_features(df["Close"], volume, leader)
                self.features_cache[ticker] = feats
            except Exception as e:
                print(f"  Warning: {ticker} feature computation failed: {e}")

        print(f"  Features computed for {len(self.features_cache)} tickers")

    def run(self, start_date, end_date, verbose=True):
        """Run the backtest over a date range."""
        self.strategy.reset()

        market_df = self.data_dict[self.market_ticker]
        all_dates = market_df.loc[start_date:end_date].index

        if verbose:
            print(f"Running backtest: {start_date} to {end_date} "
                  f"({len(all_dates)} days)")

        portfolio_values = [1.0]
        daily_returns = []

        for i, date in enumerate(all_dates):
            prices = {}
            for ticker, df in self.data_dict.items():
                if date in df.index and "Close" in df.columns:
                    prices[ticker] = df.loc[date, "Close"]

            features_dict = {}
            for ticker, feats in self.features_cache.items():
                if date in feats.index:
                    features_dict[ticker] = feats.loc[date].to_dict()

            stats = self.strategy.step(date, prices, features_dict)

            daily_ret = 0.0
            for ticker, pos in self.strategy.positions.items():
                if ticker in self.data_dict:
                    df = self.data_dict[ticker]
                    if date in df.index:
                        idx = df.index.get_loc(date)
                        if idx > 0:
                            prev = df.iloc[idx - 1]["Close"]
                            curr = df.iloc[idx]["Close"]
                            stock_ret = (curr / prev - 1) * pos.direction
                            daily_ret += stock_ret * pos.size

            daily_returns.append({"date": date, "return": daily_ret})
            portfolio_values.append(portfolio_values[-1] * (1 + daily_ret))

            if verbose and (i + 1) % 365 == 0:
                print(f"  Year {(i+1)//365}: portfolio={portfolio_values[-1]:.4f}, "
                      f"positions={stats['n_positions']}")

        returns_df = pd.DataFrame(daily_returns).set_index("date")
        trades_df = self.strategy.get_trade_log()
        stats_df = self.strategy.get_daily_stats()

        result = BacktestResult(
            returns=returns_df, trades=trades_df,
            daily_stats=stats_df,
            portfolio_values=pd.Series(portfolio_values[1:], index=all_dates),
            config=self.config,
        )

        if verbose:
            result.print_summary()

        return result


class BacktestResult:
    """Container for backtest results."""

    def __init__(self, returns, trades, daily_stats, portfolio_values, config):
        self.returns = returns
        self.trades = trades
        self.daily_stats = daily_stats
        self.portfolio_values = portfolio_values
        self.config = config

    def total_return(self):
        return self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0] - 1

    def cagr(self):
        n_years = len(self.returns) / TRADING_DAYS_PER_YEAR
        if n_years <= 0:
            return 0
        return (1 + self.total_return()) ** (1 / n_years) - 1

    def sharpe_ratio(self, risk_free_rate=0.02):
        """Annualized Sharpe ratio (365 days for crypto)."""
        excess = self.returns["return"] - risk_free_rate / TRADING_DAYS_PER_YEAR
        if excess.std() == 0:
            return 0
        return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def sortino_ratio(self, risk_free_rate=0.02):
        excess = self.returns["return"] - risk_free_rate / TRADING_DAYS_PER_YEAR
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0
        return excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def max_drawdown(self):
        cummax = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - cummax) / cummax
        return drawdown.min()

    def calmar_ratio(self):
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0
        return self.cagr() / mdd

    def win_rate(self):
        if len(self.trades) == 0:
            return 0
        return (self.trades["net_pnl"] > 0).mean()

    def profit_factor(self):
        if len(self.trades) == 0:
            return 0
        wins = self.trades.loc[self.trades["net_pnl"] > 0, "net_pnl"].sum()
        losses = abs(self.trades.loc[self.trades["net_pnl"] < 0, "net_pnl"].sum())
        if losses == 0:
            return float("inf") if wins > 0 else 0
        return wins / losses

    def avg_trade_pnl(self):
        if len(self.trades) == 0:
            return 0
        return self.trades["net_pnl"].mean()

    def avg_hold_days(self):
        if len(self.trades) == 0:
            return 0
        return self.trades["days_held"].mean()

    def n_trades(self):
        return len(self.trades)

    def annual_turnover(self):
        n_years = len(self.returns) / TRADING_DAYS_PER_YEAR
        if n_years == 0:
            return 0
        return len(self.trades) / n_years

    def print_summary(self):
        print("\n" + "=" * 60)
        print("CRYPTO BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period:         {self.returns.index[0].date()} to "
              f"{self.returns.index[-1].date()}")
        print(f"Calendar days:  {len(self.returns)}")
        print()
        print("--- RETURNS ---")
        print(f"Total return:   {self.total_return():.2%}")
        print(f"CAGR:           {self.cagr():.2%}")
        print(f"Sharpe ratio:   {self.sharpe_ratio():.3f}")
        print(f"Sortino ratio:  {self.sortino_ratio():.3f}")
        print(f"Max drawdown:   {self.max_drawdown():.2%}")
        print(f"Calmar ratio:   {self.calmar_ratio():.3f}")
        print()
        print("--- TRADES ---")
        print(f"Total trades:   {self.n_trades()}")
        print(f"Win rate:       {self.win_rate():.2%}")
        print(f"Profit factor:  {self.profit_factor():.2f}")
        print(f"Avg trade PnL:  {self.avg_trade_pnl():.4f}")
        print(f"Avg hold days:  {self.avg_hold_days():.1f}")
        print(f"Annual turnover:{self.annual_turnover():.0f} trades/yr")
        print()
        print("--- RISK ---")
        ann_vol = self.returns["return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        print(f"Ann. volatility:{ann_vol:.2%}")
        print(f"Daily VaR(5%):  {self.returns['return'].quantile(0.05):.4f}")
        q5 = self.returns['return'].quantile(0.05)
        cvar = self.returns['return'][self.returns['return'] <= q5].mean()
        print(f"Daily CVaR(5%): {cvar:.4f}")

        if len(self.trades) > 0:
            print()
            print("--- EXIT REASONS ---")
            for reason, count in self.trades["exit_reason"].apply(
                lambda x: x.split(" ")[0]
            ).value_counts().items():
                print(f"  {reason}: {count} ({count/len(self.trades):.0%})")
        print("=" * 60)

    def to_dict(self):
        return {
            "total_return": self.total_return(),
            "cagr": self.cagr(),
            "sharpe": self.sharpe_ratio(),
            "sortino": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown(),
            "calmar": self.calmar_ratio(),
            "n_trades": self.n_trades(),
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "avg_trade_pnl": self.avg_trade_pnl(),
            "avg_hold_days": self.avg_hold_days(),
            "annual_turnover": self.annual_turnover(),
        }


def run_benchmark_comparison(data_dict, result, market_ticker="BTC-USD"):
    """Compare strategy to BTC buy-and-hold."""
    market = data_dict[market_ticker]
    dates = result.returns.index

    btc_prices = market.loc[dates, "Close"]
    btc_ret = btc_prices.pct_change().fillna(0)
    btc_total = (1 + btc_ret).cumprod().iloc[-1] - 1
    btc_sharpe = btc_ret.mean() / btc_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if btc_ret.std() > 0 else 0
    btc_mdd = ((btc_prices.cummax() - btc_prices) / btc_prices.cummax()).max()

    n_days = len(dates)
    btc_cagr = (1 + btc_total) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Crypto TMD-ARC':>14} {'BTC B&H':>12}")
    print("-" * 46)
    print(f"{'Total Return':<20} {result.total_return():>14.2%} {btc_total:>12.2%}")
    print(f"{'CAGR':<20} {result.cagr():>14.2%} {btc_cagr:>12.2%}")
    print(f"{'Sharpe':<20} {result.sharpe_ratio():>14.3f} {btc_sharpe:>12.3f}")
    print(f"{'Max Drawdown':<20} {result.max_drawdown():>14.2%} {-btc_mdd:>12.2%}")
    print("=" * 60)

    return {
        "btc_total_return": btc_total,
        "btc_cagr": btc_cagr,
        "btc_sharpe": btc_sharpe,
        "btc_max_drawdown": -btc_mdd,
    }
