"""
Crypto TMD-ARC Strategy Engine
================================
Temporal Momentum Dispersion with Adaptive Regime Cascade — Crypto Edition

Crypto-specific adaptations:
- Higher volatility thresholds for regime detection
- Adjusted position sizing for crypto vol levels
- Higher transaction costs (crypto spreads + exchange fees)
- Shorter holding periods (crypto moves faster)
- 24/7 market → calendar day counting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CryptoStrategyConfig:
    """Strategy parameters adapted for crypto markets."""

    # === ENTRY SIGNALS ===
    mtmdi_zscore_entry: float = 1.5
    cacs_entry_threshold: float = 0.03  # Higher for crypto (more noise)
    mpr_threshold: float = 0.5

    # === EXIT SIGNALS ===
    mtmdi_zscore_exit: float = 0.5
    max_hold_days: int = 21  # Calendar days
    stop_loss: float = -0.10  # 10% stop (crypto is more volatile)
    take_profit: float = 0.35  # 35% take profit

    # === POSITION SIZING ===
    max_position_pct: float = 0.08  # 8% per coin (concentrated crypto portfolio)
    max_total_exposure: float = 0.80
    vol_target: float = 0.25  # Higher vol target for crypto

    # === REGIME ADAPTATION ===
    high_vol_reduction: float = 0.4  # More aggressive reduction in high vol
    low_vol_boost: float = 1.0

    # === TRANSACTION COSTS ===
    transaction_cost_bps: float = 15  # 15 bps (crypto exchanges typically charge more)


@dataclass
class Signal:
    """A trading signal."""
    ticker: str
    date: pd.Timestamp
    direction: int
    strength: float
    mtmdi_zscore: float
    cascade_gap: float
    mpr: float
    vol_regime: str
    rationale: str


@dataclass
class Position:
    """An open position."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    direction: int
    size: float
    strength: float
    days_held: int = 0
    pnl: float = 0.0


class CryptoTMDArcStrategy:
    """
    TMD-ARC strategy adapted for cryptocurrency markets.
    LONG-ONLY strategy for crypto.
    """

    def __init__(self, config: Optional[CryptoStrategyConfig] = None):
        self.config = config or CryptoStrategyConfig()
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []
        self.signals_log: list[Signal] = []
        self.daily_pnl: list[dict] = []

    def detect_vol_regime(self, vol_ratio_7_30: float, vol_30d: float) -> str:
        """
        Crypto regime classification — higher thresholds than stocks.
        Crypto vol_30d of 0.60+ is "high" (stocks: 0.30+)
        """
        if vol_ratio_7_30 > 1.5 or vol_30d > 0.60:
            return "high"
        elif vol_ratio_7_30 < 0.7 and vol_30d < 0.25:
            return "low"
        else:
            return "normal"

    def compute_position_size(self, strength: float, vol_30d: float,
                               vol_regime: str) -> float:
        """Volatility-targeted position sizing for crypto."""
        cfg = self.config
        if vol_30d < 0.05:
            vol_30d = 0.05
        base_size = cfg.vol_target / vol_30d
        size = base_size * strength

        if vol_regime == "high":
            size *= cfg.high_vol_reduction
        elif vol_regime == "low":
            size *= cfg.low_vol_boost

        size = min(size, cfg.max_position_pct)
        return size

    def generate_signals(self, date, features_dict):
        """Generate signals for all coins on a given date."""
        signals = []
        cfg = self.config

        for ticker, feat in features_dict.items():
            if ticker in self.positions:
                continue

            mtmdi_z = feat.get("mtmdi_zscore", 0)
            mtmdi_dir = feat.get("mtmdi_direction", 0)
            cascade = feat.get("cacs", 0)
            mpr = feat.get("mpr_zscore", 0)
            vol_ratio = feat.get("vol_ratio_7_30", 1.0)
            vol_30d = feat.get("vol_30d", 0.40)

            if any(np.isnan(v) for v in [mtmdi_z, mtmdi_dir, vol_30d]):
                continue

            if abs(mtmdi_z) < cfg.mtmdi_zscore_entry:
                continue
            if mtmdi_dir <= 0:
                continue

            cascade_val = cascade if not np.isnan(cascade) else 0
            mpr_val = mpr if not np.isnan(mpr) else 0

            has_cascade = cascade_val > cfg.cacs_entry_threshold
            has_momentum = mpr_val > cfg.mpr_threshold

            if not (has_cascade or has_momentum):
                continue

            strength = (
                min(abs(mtmdi_z) / 3.0, 1.0) * 0.5 +
                min(abs(cascade_val) / 0.08, 1.0) * 0.3 +
                min(max(mpr_val, 0) / 2.0, 1.0) * 0.2
            )

            vol_regime = self.detect_vol_regime(vol_ratio, vol_30d)

            rationale_parts = [f"MTMDI z={mtmdi_z:.2f}"]
            if has_cascade:
                rationale_parts.append(f"cascade={cascade_val:.3f}")
            if has_momentum:
                rationale_parts.append(f"MPR z={mpr_val:.2f}")
            rationale_parts.append(f"regime={vol_regime}")

            signal = Signal(
                ticker=ticker, date=date, direction=1,
                strength=strength, mtmdi_zscore=mtmdi_z,
                cascade_gap=cascade_val, mpr=mpr_val,
                vol_regime=vol_regime,
                rationale="; ".join(rationale_parts),
            )
            signals.append(signal)

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals

    def update_positions(self, date, prices, features_dict):
        """Update existing positions: check exits, update PnL."""
        cfg = self.config
        closed = []

        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            pos.days_held += 1

            current_price = prices.get(ticker)
            if current_price is None or np.isnan(current_price):
                continue

            pos.pnl = (current_price / pos.entry_price - 1) * pos.direction

            should_exit = False
            exit_reason = ""

            if pos.pnl <= cfg.stop_loss:
                should_exit = True
                exit_reason = f"stop_loss ({pos.pnl:.2%})"
            elif pos.pnl >= cfg.take_profit:
                should_exit = True
                exit_reason = f"take_profit ({pos.pnl:.2%})"
            elif pos.days_held >= cfg.max_hold_days:
                should_exit = True
                exit_reason = f"max_hold ({pos.days_held}d)"
            elif ticker in features_dict:
                feat = features_dict[ticker]
                mtmdi_z = feat.get("mtmdi_zscore", 0)
                if not np.isnan(mtmdi_z) and abs(mtmdi_z) < cfg.mtmdi_zscore_exit:
                    should_exit = True
                    exit_reason = f"mtmdi_resolved (z={mtmdi_z:.2f})"

            if should_exit:
                tc = cfg.transaction_cost_bps / 10000
                net_pnl = pos.pnl - 2 * tc

                trade = {
                    "ticker": ticker,
                    "entry_date": pos.entry_date,
                    "exit_date": date,
                    "entry_price": pos.entry_price,
                    "exit_price": current_price,
                    "direction": pos.direction,
                    "size": pos.size,
                    "gross_pnl": pos.pnl,
                    "net_pnl": net_pnl,
                    "days_held": pos.days_held,
                    "exit_reason": exit_reason,
                }
                closed.append(trade)
                del self.positions[ticker]

        return closed

    def execute_signals(self, signals, prices, features_dict):
        """Open new positions from signals."""
        cfg = self.config
        total_exposure = sum(p.size for p in self.positions.values())

        for signal in signals:
            if total_exposure >= cfg.max_total_exposure:
                break

            price = prices.get(signal.ticker)
            if price is None or np.isnan(price):
                continue

            feat = features_dict.get(signal.ticker, {})
            vol_30d = feat.get("vol_30d", 0.40)
            if np.isnan(vol_30d):
                vol_30d = 0.40

            size = self.compute_position_size(
                signal.strength, vol_30d, signal.vol_regime
            )
            remaining = cfg.max_total_exposure - total_exposure
            size = min(size, remaining)

            if size < 0.005:
                continue

            self.positions[signal.ticker] = Position(
                ticker=signal.ticker, entry_date=signal.date,
                entry_price=price, direction=signal.direction,
                size=size, strength=signal.strength,
            )
            total_exposure += size
            self.signals_log.append(signal)

    def step(self, date, prices, features_dict):
        """Execute one day of the strategy."""
        closed = self.update_positions(date, prices, features_dict)
        self.closed_trades.extend(closed)

        signals = self.generate_signals(date, features_dict)
        self.execute_signals(signals, prices, features_dict)

        total_exposure = sum(p.size for p in self.positions.values())
        portfolio_pnl = sum(p.pnl * p.size for p in self.positions.values())

        daily_stats = {
            "date": date,
            "n_positions": len(self.positions),
            "total_exposure": total_exposure,
            "portfolio_pnl": portfolio_pnl,
            "n_signals": len(signals),
            "n_closed": len(closed),
        }
        self.daily_pnl.append(daily_stats)
        return daily_stats

    def get_trade_log(self):
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_trades)

    def get_daily_stats(self):
        if not self.daily_pnl:
            return pd.DataFrame()
        df = pd.DataFrame(self.daily_pnl)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def reset(self):
        self.positions = {}
        self.closed_trades = []
        self.signals_log = []
        self.daily_pnl = []
