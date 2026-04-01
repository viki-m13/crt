#!/usr/bin/env python3
"""
stock_pred_rl_env.py — RL Environment for Stock Selection
==========================================================
Frames daily stock selection as a Markov Decision Process.

STATE: For each stock on a given day, the agent sees its normalized features.
       The agent processes stocks sequentially and selects the top-K to buy.

ACTION: Binary — buy (1) or skip (0) for the current stock candidate.

REWARD: Based on the 30-day forward return of selected stocks.
        +1 if the stock gains >= 10% (hit target)
        -0.5 if the stock loses money (penalty for bad picks)
        +0.2 for modest gains (0-10%)
        -0.1 for skipping a stock that would have hit target (opportunity cost)

ANTI-LEAKAGE:
- Environment is initialized with a DATE RANGE — it can only see features
  computed from data BEFORE each decision date
- Forward returns are used ONLY for reward computation AFTER the action
- The agent never sees forward returns as part of the state
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class StockSelectionEnv(gym.Env):
    """
    RL environment for selecting stocks likely to gain 10%+ in 30 days.

    The agent processes one stock-day observation at a time and decides
    whether to "buy" (predict it will hit +10%) or "skip".
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset, feature_cols, date_range=None, max_daily_picks=3):
        """
        Args:
            dataset: DataFrame with features, labels, metadata
            feature_cols: list of feature column names
            date_range: (start_date, end_date) tuple to restrict environment
            max_daily_picks: max stocks to pick per day
        """
        super().__init__()

        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        self.max_daily_picks = max_daily_picks

        # Filter to date range
        if date_range is not None:
            mask = (
                (pd.to_datetime(dataset["date"]) >= pd.Timestamp(date_range[0])) &
                (pd.to_datetime(dataset["date"]) <= pd.Timestamp(date_range[1]))
            )
            self.dataset = dataset[mask].reset_index(drop=True)
        else:
            self.dataset = dataset.reset_index(drop=True)

        # Group by date for sequential processing
        self.dataset["_date_str"] = pd.to_datetime(self.dataset["date"]).dt.strftime("%Y-%m-%d")
        self.dates = sorted(self.dataset["_date_str"].unique())

        # State: features + context (picks_today / max_picks, day_progress)
        self.observation_space = spaces.Box(
            low=-6.0, high=6.0,
            shape=(self.n_features + 2,),
            dtype=np.float32,
        )

        # Action: 0 = skip, 1 = buy
        self.action_space = spaces.Discrete(2)

        # Episode tracking
        self.current_date_idx = 0
        self.current_stock_idx = 0
        self.picks_today = 0
        self.episode_stocks = []
        self.episode_rewards = []

    def _get_daily_stocks(self, date_str):
        """Get all stock observations for a given date."""
        mask = self.dataset["_date_str"] == date_str
        return self.dataset[mask].reset_index(drop=True)

    def _get_obs(self):
        """Get current observation (features + context)."""
        date_str = self.dates[self.current_date_idx]
        daily = self._get_daily_stocks(date_str)

        if self.current_stock_idx >= len(daily):
            # Shouldn't happen, but return zeros
            return np.zeros(self.n_features + 2, dtype=np.float32)

        row = daily.iloc[self.current_stock_idx]
        features = row[self.feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        features = np.clip(features, -5.0, 5.0)

        # Context features
        picks_ratio = self.picks_today / self.max_daily_picks
        day_progress = self.current_stock_idx / max(len(daily), 1)

        obs = np.concatenate([features, [picks_ratio, day_progress]])
        return obs.astype(np.float32)

    def _compute_reward(self, action, row):
        """
        Compute reward based on action and actual outcome.
        Reward shaping to encourage finding +10% stocks.
        """
        fwd_return = row["fwd_return_30d"]
        is_winner = row["label"] == 1

        if action == 1:  # BUY
            if is_winner:
                return 1.0  # correct buy — hit target
            elif fwd_return >= 0:
                return 0.1  # ok, at least didn't lose money
            else:
                return -0.5  # bad pick — lost money
        else:  # SKIP
            if is_winner:
                return -0.05  # small penalty for missing a winner
            else:
                return 0.0  # correctly avoided

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_date_idx = 0
        self.current_stock_idx = 0
        self.picks_today = 0
        self.episode_stocks = []
        self.episode_rewards = []
        return self._get_obs(), {}

    def step(self, action):
        date_str = self.dates[self.current_date_idx]
        daily = self._get_daily_stocks(date_str)

        if self.current_stock_idx >= len(daily):
            # End of day — move to next day
            return self._advance_day()

        row = daily.iloc[self.current_stock_idx]

        # Enforce max daily picks
        if action == 1 and self.picks_today >= self.max_daily_picks:
            action = 0  # forced skip

        reward = self._compute_reward(action, row)

        if action == 1:
            self.picks_today += 1
            self.episode_stocks.append({
                "date": date_str,
                "ticker": row["ticker"],
                "fwd_return": row["fwd_return_30d"],
                "label": row["label"],
            })

        self.episode_rewards.append(reward)
        self.current_stock_idx += 1

        # Check if we've processed all stocks for today
        if self.current_stock_idx >= len(daily):
            return self._advance_day()

        obs = self._get_obs()
        return obs, reward, False, False, {}

    def _advance_day(self):
        """Move to next trading day."""
        self.current_date_idx += 1
        self.current_stock_idx = 0
        self.picks_today = 0

        terminated = self.current_date_idx >= len(self.dates)

        if terminated:
            obs = np.zeros(self.n_features + 2, dtype=np.float32)
            return obs, 0.0, True, False, self._get_episode_info()

        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def _get_episode_info(self):
        """Summary info at episode end."""
        if not self.episode_stocks:
            return {"n_picks": 0, "avg_return": 0, "hit_rate": 0}

        picks = pd.DataFrame(self.episode_stocks)
        return {
            "n_picks": len(picks),
            "avg_return": float(picks["fwd_return"].mean()),
            "hit_rate": float(picks["label"].mean()),
            "total_reward": float(sum(self.episode_rewards)),
        }


class StockRankingEnv(gym.Env):
    """
    Alternative RL formulation: rank stocks by predicted upside.

    STATE: All stock features for one day (flattened or padded)
    ACTION: Continuous scores for each stock — top-K are selected
    REWARD: Average return of selected stocks

    This is better suited for the actual task but requires more complex
    action spaces. Uses a fixed-size observation for compatibility.
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset, feature_cols, date_range=None,
                 max_stocks_per_day=100, top_k=3):
        super().__init__()

        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        self.max_stocks = max_stocks_per_day
        self.top_k = top_k

        if date_range is not None:
            mask = (
                (pd.to_datetime(dataset["date"]) >= pd.Timestamp(date_range[0])) &
                (pd.to_datetime(dataset["date"]) <= pd.Timestamp(date_range[1]))
            )
            self.dataset = dataset[mask].reset_index(drop=True)
        else:
            self.dataset = dataset.reset_index(drop=True)

        self.dataset["_date_str"] = pd.to_datetime(self.dataset["date"]).dt.strftime("%Y-%m-%d")
        self.dates = sorted(self.dataset["_date_str"].unique())

        # State: padded matrix of stock features (max_stocks x n_features), flattened
        # + mask indicating which slots are real stocks
        obs_size = self.max_stocks * self.n_features + self.max_stocks
        self.observation_space = spaces.Box(
            low=-6.0, high=6.0, shape=(obs_size,), dtype=np.float32
        )

        # Action: score for each stock slot
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_stocks,), dtype=np.float32
        )

        self.current_day = 0

    def _get_obs(self):
        if self.current_day >= len(self.dates):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        date_str = self.dates[self.current_day]
        daily = self.dataset[self.dataset["_date_str"] == date_str]

        n_stocks = min(len(daily), self.max_stocks)
        features = np.zeros((self.max_stocks, self.n_features), dtype=np.float32)
        mask = np.zeros(self.max_stocks, dtype=np.float32)

        feat_vals = daily[self.feature_cols].values[:n_stocks].astype(np.float32)
        feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=5.0, neginf=-5.0)
        feat_vals = np.clip(feat_vals, -5.0, 5.0)

        features[:n_stocks] = feat_vals
        mask[:n_stocks] = 1.0

        obs = np.concatenate([features.flatten(), mask])
        return obs

    def _get_daily_data(self):
        date_str = self.dates[self.current_day]
        return self.dataset[self.dataset["_date_str"] == date_str].reset_index(drop=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.current_day >= len(self.dates):
            return self._get_obs(), 0.0, True, False, {}

        daily = self._get_daily_data()
        n_stocks = min(len(daily), self.max_stocks)

        # Select top-K stocks by action scores (only among real stocks)
        scores = action[:n_stocks]
        top_indices = np.argsort(scores)[-self.top_k:]

        selected = daily.iloc[top_indices]
        avg_return = selected["fwd_return_30d"].mean()
        hit_rate = selected["label"].mean()

        # Reward: emphasize finding 10%+ gainers
        reward = float(hit_rate * 2.0 + avg_return * 5.0)

        self.current_day += 1
        terminated = self.current_day >= len(self.dates)

        obs = self._get_obs()
        info = {
            "avg_return": float(avg_return),
            "hit_rate": float(hit_rate),
            "n_selected": len(selected),
            "selected_tickers": selected["ticker"].tolist(),
        }

        return obs, reward, terminated, False, info
