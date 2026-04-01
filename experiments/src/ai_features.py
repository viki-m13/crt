"""
AI-Enhanced Feature Engine for CCRL Stock Prediction
=====================================================
PROPRIETARY / PATENTABLE

Novel features beyond classical technical analysis:

1. Temporal Attention Momentum (TAM)
   - Attention-weighted momentum that learns which lookback periods
     matter most in the current regime
   - Uses softmax attention over multi-scale returns
   - Patent claim: Dynamic weighting of momentum signals via learned
     attention mechanism that adapts to market regime

2. Cascade Propagation Features (CPF)
   - Models information flow across a network of related assets
   - Detects which stocks lead and which lag in real-time
   - Patent claim: Graph-based information propagation scoring for
     cross-asset momentum prediction

3. Microstructure Regime Fingerprint (MRF)
   - Encodes the current market microstructure state as a dense vector
   - Uses rolling PCA on a matrix of cross-asset features
   - Patent claim: Compressed regime representation via rolling PCA
     on multi-asset feature matrices

4. Asymmetric Tail Features (ATF)
   - Specifically engineered to predict 10%+ upside moves
   - Measures the shape of the return distribution tail
   - Patent claim: Tail-shape features for asymmetric return prediction

IMPORTANT: All features use ONLY past data. No lookahead bias.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. TEMPORAL ATTENTION MOMENTUM (TAM)
# ============================================================

def temporal_attention_momentum(close, windows=None, attention_lookback=63):
    """
    Temporal Attention Momentum (TAM)
    ==================================
    NOVEL PATENTABLE FEATURE

    Instead of equal-weighting momentum across timeframes (as in traditional
    multi-factor models), TAM uses a softmax attention mechanism to dynamically
    weight momentum signals based on which timeframes have been most predictive
    in the recent past.

    Method:
    1. Compute momentum returns at multiple timeframes
    2. For each timeframe, compute its recent "predictive score" —
       the rank correlation between the momentum signal and subsequent
       5-day forward returns over the past `attention_lookback` days
    3. Apply softmax to predictive scores to get attention weights
    4. TAM = weighted sum of momentum signals using attention weights

    This is fundamentally different from:
    - Simple momentum (uses only one timeframe)
    - Equal-weighted multi-factor (static weights)
    - Optimized weights (overfit to training data)

    TAM adapts IN REAL TIME to which timeframes matter, with NO optimization.

    Returns DataFrame with:
    - tam: the attention-weighted momentum score
    - tam_entropy: entropy of attention weights (low = concentrated, high = diffuse)
    - tam_dominant_window: which timeframe currently has highest attention
    - tam_attention_*: individual attention weights for each window
    """
    if windows is None:
        windows = [5, 10, 21, 42, 63, 126, 252]

    # Step 1: Compute momentum returns at each timeframe
    mom_returns = pd.DataFrame(index=close.index)
    for w in windows:
        mom_returns[f"mom_{w}"] = close.pct_change(w)

    # Step 2: Compute 5-day forward returns (for attention scoring ONLY)
    # CRITICAL: We use PAST forward returns to score attention, not future ones.
    # At time t, we look at how well each momentum signal predicted returns
    # from t-attention_lookback to t-5 (all fully realized, no leakage).
    fwd_5d = close.pct_change(5).shift(-5)  # This is the "realized" target

    # Step 3: Rolling rank correlation of each momentum with realized fwd returns
    # We shift fwd_5d by +5 to align: at time t, we correlate
    # mom(t-lookback:t-5) with fwd_5d(t-lookback:t-5) — all past data
    attention_scores = pd.DataFrame(index=close.index)
    for w in windows:
        col = f"mom_{w}"
        # Rolling rank correlation using ONLY past realized data
        # At time t: correlate mom[t-lookback-5 : t-5] with fwd_5d[t-lookback-5 : t-5]
        # Both are fully known at time t (no leakage)
        mom_past = mom_returns[col].shift(5)  # mom at t-5
        fwd_past = fwd_5d.shift(5)            # fwd return realized by t

        def _rolling_spearman(s1, s2, window):
            """Compute rolling Spearman correlation."""
            result = pd.Series(index=s1.index, dtype=float)
            for i in range(window, len(s1)):
                x = s1.iloc[i-window:i].values
                y = s2.iloc[i-window:i].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() >= window // 2:
                    r, _ = scipy_stats.spearmanr(x[mask], y[mask])
                    result.iloc[i] = r
            return result

        # Use a vectorized approximation for speed
        corr = mom_past.rolling(attention_lookback, min_periods=attention_lookback//2).corr(fwd_past)
        attention_scores[f"attn_{w}"] = corr.clip(-1, 1)

    # Step 4: Softmax attention weights
    # Temperature parameter controls how peaked the attention is
    temperature = 2.0
    attn_cols = [f"attn_{w}" for w in windows]

    # Fill NaN with 0 (no information = no preference)
    scores_filled = attention_scores[attn_cols].fillna(0)

    # Softmax: exp(score/T) / sum(exp(score/T))
    exp_scores = np.exp(scores_filled / temperature)
    weight_sum = exp_scores.sum(axis=1).clip(lower=1e-10)
    attention_weights = exp_scores.div(weight_sum, axis=0)

    # Step 5: Compute TAM as attention-weighted momentum
    tam = pd.Series(0.0, index=close.index)
    for i, w in enumerate(windows):
        tam += attention_weights[f"attn_{w}"] * mom_returns[f"mom_{w}"].fillna(0)

    # Step 6: Compute attention entropy (information measure)
    # High entropy = all timeframes equally important (uncertain regime)
    # Low entropy = one timeframe dominates (clear regime)
    weights_arr = attention_weights.values
    weights_arr = np.clip(weights_arr, 1e-10, 1.0)
    entropy = -np.sum(weights_arr * np.log(weights_arr), axis=1)
    max_entropy = np.log(len(windows))

    # Dominant window
    dominant_idx = attention_weights.values.argmax(axis=1)
    dominant_window = pd.Series(
        [windows[i] if not np.isnan(attention_weights.iloc[j].sum())
         else np.nan for j, i in enumerate(dominant_idx)],
        index=close.index
    )

    result = pd.DataFrame({
        "tam": tam,
        "tam_entropy": entropy / max_entropy,  # Normalized to [0, 1]
        "tam_dominant_window": dominant_window,
    }, index=close.index)

    # Add individual attention weights
    for w in windows:
        result[f"tam_attn_{w}d"] = attention_weights[f"attn_{w}"]

    return result


# ============================================================
# 2. CASCADE PROPAGATION FEATURES (CPF)
# ============================================================

def cascade_propagation_features(stock_close, peer_closes, lookback=21):
    """
    Cascade Propagation Features (CPF)
    ====================================
    NOVEL PATENTABLE FEATURE

    Models information propagation across a network of related assets.
    Instead of just comparing to one leader (like CACS), this models
    the FULL propagation network:

    1. For each peer, compute lead-lag relationship at multiple lags
    2. Identify which peers are "information leaders" for this stock
    3. Compute the "cascade gap" — how much unrealized information
       exists from leaders' recent moves
    4. Score the "cascade velocity" — is the gap closing or widening?

    Patent claim: Multi-asset cascade propagation scoring that identifies
    information leaders dynamically and predicts follower catch-up timing.

    Parameters:
    - stock_close: pd.Series of the target stock's close prices
    - peer_closes: dict of {ticker: pd.Series} for peer stocks
    - lookback: window for computing correlations

    Returns DataFrame with:
    - cpf_leader_count: how many peers lead this stock
    - cpf_cascade_gap: total unrealized information from leaders
    - cpf_cascade_velocity: rate of change of cascade gap
    - cpf_network_centrality: is this stock a leader or follower overall
    """
    if not peer_closes:
        return pd.DataFrame(index=stock_close.index)

    stock_ret = stock_close.pct_change()

    leader_scores = []
    cascade_gaps = []

    for ticker, peer_close in peer_closes.items():
        peer_ret = peer_close.pct_change()

        # Align
        common = stock_ret.index.intersection(peer_ret.index)
        if len(common) < lookback * 2:
            continue

        sr = stock_ret.reindex(common)
        pr = peer_ret.reindex(common)

        # Lead-lag analysis: does peer lead stock?
        # Correlation of peer_ret(t-1) with stock_ret(t)
        lag1_corr = pr.shift(1).rolling(lookback, min_periods=lookback//2).corr(sr)
        # Correlation of peer_ret(t) with stock_ret(t)
        contemp_corr = pr.rolling(lookback, min_periods=lookback//2).corr(sr)

        # "Lead score" = lagged_corr - contemporaneous_corr
        # Positive = peer leads stock (information propagates with delay)
        lead_score = (lag1_corr - contemp_corr).reindex(stock_close.index)
        leader_scores.append(lead_score)

        # Cascade gap: peer's recent move that stock hasn't responded to yet
        peer_move = peer_close.pct_change(lookback).reindex(common)
        stock_move = stock_close.pct_change(lookback).reindex(common)

        # Rolling beta
        cov = sr.rolling(lookback, min_periods=lookback//2).cov(pr)
        var = pr.rolling(lookback, min_periods=lookback//2).var().clip(lower=1e-10)
        beta = cov / var

        expected = peer_move * beta
        gap = (expected - stock_move).reindex(stock_close.index)
        # Weight gap by lead score (only count gaps from actual leaders)
        weighted_gap = gap * lead_score.clip(lower=0)
        cascade_gaps.append(weighted_gap)

    if not leader_scores:
        return pd.DataFrame({
            "cpf_leader_count": 0,
            "cpf_cascade_gap": 0.0,
            "cpf_cascade_velocity": 0.0,
            "cpf_network_centrality": 0.0,
        }, index=stock_close.index)

    leader_df = pd.DataFrame(leader_scores).T
    gap_df = pd.DataFrame(cascade_gaps).T

    # Number of peers that lead this stock (lead_score > threshold)
    leader_count = (leader_df > 0.05).sum(axis=1)

    # Total cascade gap from all leaders
    total_gap = gap_df.sum(axis=1)

    # Cascade velocity (is the gap closing or widening?)
    cascade_velocity = total_gap - total_gap.shift(5)

    # Network centrality: mean lead score (negative = this stock is a leader)
    centrality = leader_df.mean(axis=1)

    return pd.DataFrame({
        "cpf_leader_count": leader_count,
        "cpf_cascade_gap": total_gap,
        "cpf_cascade_velocity": cascade_velocity,
        "cpf_network_centrality": centrality,
    }, index=stock_close.index)


# ============================================================
# 3. MICROSTRUCTURE REGIME FINGERPRINT (MRF)
# ============================================================

def microstructure_regime_fingerprint(close_dict, n_components=5, lookback=63):
    """
    Microstructure Regime Fingerprint (MRF)
    =========================================
    NOVEL PATENTABLE FEATURE

    Compresses the current state of the entire market into a dense
    regime vector using rolling PCA on cross-asset features.

    Unlike simple regime detection (e.g., VIX high/low), this captures
    the FULL structure of cross-asset relationships:
    - Correlation structure changes
    - Volatility term structure shifts
    - Cross-sector rotation patterns
    - Flight-to-quality flows

    Patent claim: Rolling PCA-based regime fingerprinting that creates
    a continuous, multi-dimensional regime representation from cross-asset
    feature matrices, enabling regime-conditional stock selection.

    Parameters:
    - close_dict: {ticker: pd.Series of close prices}
    - n_components: number of PCA components to retain
    - lookback: rolling window for PCA

    Returns: dict of {ticker: DataFrame with MRF features}
    """
    # Build cross-asset return matrix
    tickers = sorted(close_dict.keys())
    if len(tickers) < n_components + 2:
        return {}

    # Compute returns for all assets
    returns_df = pd.DataFrame({
        t: close_dict[t].pct_change() for t in tickers
    })
    # Also compute volatility for all assets
    vol_df = pd.DataFrame({
        t: close_dict[t].pct_change().rolling(21).std() for t in tickers
    })

    # Stack returns and vol into a combined feature matrix
    combined = pd.concat([returns_df, vol_df], axis=1, keys=["ret", "vol"])
    common_idx = combined.dropna().index

    if len(common_idx) < lookback * 2:
        return {}

    # Rolling PCA to extract regime fingerprint
    n_features = combined.shape[1]
    n_comp = min(n_components, n_features - 1, len(tickers) - 1)

    # Pre-allocate result arrays
    fingerprints = np.full((len(common_idx), n_comp), np.nan)
    explained_var = np.full(len(common_idx), np.nan)

    scaler = StandardScaler()

    for i in range(lookback, len(common_idx)):
        window_data = combined.iloc[i-lookback:i].values
        # Remove any remaining NaN columns
        valid_cols = ~np.any(np.isnan(window_data), axis=0)
        if valid_cols.sum() < n_comp + 1:
            continue

        window_clean = window_data[:, valid_cols]

        try:
            scaled = scaler.fit_transform(window_clean)
            pca = PCA(n_components=n_comp)
            transformed = pca.fit_transform(scaled)

            # The fingerprint is the LAST row (current state projected)
            fingerprints[i] = transformed[-1]
            explained_var[i] = pca.explained_variance_ratio_.sum()
        except Exception:
            continue

    # Build result DataFrame
    mrf_df = pd.DataFrame(
        fingerprints,
        index=common_idx,
        columns=[f"mrf_pc{i+1}" for i in range(n_comp)]
    )
    mrf_df["mrf_explained_var"] = explained_var

    # Compute regime change score (L2 distance between consecutive fingerprints)
    diffs = mrf_df[[f"mrf_pc{i+1}" for i in range(n_comp)]].diff()
    mrf_df["mrf_regime_change"] = np.sqrt((diffs**2).sum(axis=1))

    # Return as dict keyed by ticker (same MRF for all stocks)
    result = {}
    for ticker in tickers:
        result[ticker] = mrf_df.reindex(close_dict[ticker].index)

    return result


# ============================================================
# 4. ASYMMETRIC TAIL FEATURES (ATF)
# ============================================================

def asymmetric_tail_features(close, lookback=252):
    """
    Asymmetric Tail Features (ATF)
    ================================
    NOVEL PATENTABLE FEATURE

    Specifically engineered to predict 10%+ upside moves within 30 days.

    Traditional features (momentum, volatility) are symmetric — they don't
    distinguish between upside and downside potential. ATF explicitly models
    the SHAPE of the return distribution's right tail.

    Features:
    1. Right-tail thickness: How often has this stock made 10%+ moves
       in the past? Stocks with fat right tails do it more often.
    2. Coiled spring score: Low recent vol + high historical vol range
       suggests energy building for a large move.
    3. Positive skew momentum: Is the distribution becoming more
       right-skewed? This precedes large upside moves.
    4. Tail acceleration: Is the right tail getting fatter over time?
    5. Drawdown recovery propensity: How quickly does this stock recover
       from drawdowns? Fast recoverers are more likely to make 10%+ moves.

    Patent claim: Asymmetric tail-shape feature extraction for predicting
    large positive return events, combining distribution shape analysis
    with regime-conditional tail thickness estimation.
    """
    log_ret = np.log(close / close.shift(1))
    daily_ret = close.pct_change()

    # 1. Right-tail thickness (historical frequency of 10%+ monthly moves)
    monthly_ret = close.pct_change(21)
    right_tail_freq = monthly_ret.rolling(lookback, min_periods=lookback//2).apply(
        lambda x: (x >= 0.10).mean(), raw=True
    )

    # 2. Coiled spring score
    # Recent vol (21d) vs long-term vol range
    vol_21 = log_ret.rolling(21).std() * np.sqrt(252)
    vol_252_high = vol_21.rolling(lookback, min_periods=lookback//2).max()
    vol_252_low = vol_21.rolling(lookback, min_periods=lookback//2).min()
    vol_range = (vol_252_high - vol_252_low).clip(lower=1e-6)
    # Low current vol relative to range = coiled spring
    coiled_spring = 1 - (vol_21 - vol_252_low) / vol_range
    coiled_spring = coiled_spring.clip(0, 1)

    # 3. Positive skew momentum
    def _rolling_skew(x):
        if len(x) < 20:
            return 0
        return scipy_stats.skew(x[~np.isnan(x)]) if (~np.isnan(x)).sum() >= 20 else 0

    skew_63 = log_ret.rolling(63, min_periods=42).apply(_rolling_skew, raw=True)
    skew_252 = log_ret.rolling(252, min_periods=126).apply(_rolling_skew, raw=True)
    # Positive skew momentum = recent skew improving relative to long-term
    skew_momentum = skew_63 - skew_252

    # 4. Tail acceleration
    # Compare right-tail frequency in recent window vs older window
    right_tail_recent = monthly_ret.rolling(63, min_periods=42).apply(
        lambda x: (x >= 0.10).mean(), raw=True
    )
    right_tail_old = monthly_ret.rolling(lookback, min_periods=lookback//2).apply(
        lambda x: (x >= 0.10).mean(), raw=True
    )
    tail_acceleration = right_tail_recent - right_tail_old

    # 5. Drawdown recovery propensity
    rolling_max = close.rolling(lookback, min_periods=63).max()
    drawdown = (close - rolling_max) / rolling_max

    # Average speed of recovery from >10% drawdowns
    in_drawdown = drawdown < -0.10
    # Recovery rate: how quickly does drawdown improve when in drawdown?
    dd_improvement = drawdown.diff(5).clip(lower=0)  # Only count improvements
    recovery_rate = dd_improvement.rolling(lookback, min_periods=63).mean()

    # 6. Upside capture ratio vs market proxy (self-relative)
    # Ratio of positive returns to negative returns magnitude
    pos_ret = daily_ret.clip(lower=0)
    neg_ret = daily_ret.clip(upper=0).abs()
    upside_capture = (
        pos_ret.rolling(63, min_periods=42).mean() /
        neg_ret.rolling(63, min_periods=42).mean().clip(lower=1e-8)
    )

    # 7. Breakout proximity
    # How close is the stock to its 252-day high? Near highs -> breakout potential
    high_252 = close.rolling(lookback, min_periods=63).max()
    breakout_proximity = close / high_252

    return pd.DataFrame({
        "atf_right_tail_freq": right_tail_freq,
        "atf_coiled_spring": coiled_spring,
        "atf_skew_momentum": skew_momentum,
        "atf_tail_acceleration": tail_acceleration,
        "atf_recovery_propensity": recovery_rate,
        "atf_upside_capture": upside_capture,
        "atf_breakout_proximity": breakout_proximity,
    }, index=close.index)


# ============================================================
# 5. COMPOSITE FEATURE BUILDER
# ============================================================

def compute_ccrl_features(stock_close, stock_volume, market_close,
                          peer_closes=None, mrf_cache=None):
    """
    Compute the full CCRL feature set for one stock.

    Combines:
    - Temporal Attention Momentum (TAM)
    - Cascade Propagation Features (CPF)
    - Microstructure Regime Fingerprint (MRF) — passed from cache
    - Asymmetric Tail Features (ATF)
    - Classic features from the base system (momentum, vol, drawdown)

    All features are strictly backward-looking. No leakage.

    Parameters:
    - stock_close: pd.Series
    - stock_volume: pd.Series (can be None)
    - market_close: pd.Series (SPY)
    - peer_closes: dict of {ticker: pd.Series} for peer stocks
    - mrf_cache: pre-computed MRF DataFrame for this stock (or None)

    Returns: pd.DataFrame with all features
    """
    features = []

    # 1. Temporal Attention Momentum
    tam = temporal_attention_momentum(stock_close)
    features.append(tam)

    # 2. Asymmetric Tail Features
    atf = asymmetric_tail_features(stock_close)
    features.append(atf)

    # 3. Cascade Propagation Features
    if peer_closes:
        cpf = cascade_propagation_features(stock_close, peer_closes)
        features.append(cpf)

    # 4. MRF (pre-computed for efficiency)
    if mrf_cache is not None:
        features.append(mrf_cache)

    # 5. Classic momentum features (multi-scale)
    for w in [5, 10, 21, 63, 126, 252]:
        features.append(pd.DataFrame({
            f"ret_{w}d": stock_close.pct_change(w)
        }, index=stock_close.index))

    # 6. Volatility features
    log_ret = np.log(stock_close / stock_close.shift(1))
    vol_feats = pd.DataFrame({
        "vol_5d": log_ret.rolling(5).std() * np.sqrt(252),
        "vol_21d": log_ret.rolling(21).std() * np.sqrt(252),
        "vol_63d": log_ret.rolling(63).std() * np.sqrt(252),
    }, index=stock_close.index)
    vol_feats["vol_ratio_5_21"] = vol_feats["vol_5d"] / vol_feats["vol_21d"].clip(lower=1e-8)
    features.append(vol_feats)

    # 7. Drawdown features
    rolling_max = stock_close.rolling(252, min_periods=21).max()
    dd = (stock_close - rolling_max) / rolling_max
    rolling_min = stock_close.rolling(252, min_periods=21).min()
    features.append(pd.DataFrame({
        "drawdown_252d": dd,
        "dd_change_5d": dd - dd.shift(5),
        "position_in_52w_range": (stock_close - rolling_min) / (rolling_max - rolling_min).clip(lower=1e-8),
    }, index=stock_close.index))

    # 8. Volume features
    if stock_volume is not None:
        vol_ma20 = stock_volume.rolling(20).mean().clip(lower=1)
        features.append(pd.DataFrame({
            "volume_relative": stock_volume / vol_ma20,
            "volume_trend": stock_volume.rolling(5).mean() / vol_ma20,
        }, index=stock_close.index))

    # 9. Market-relative features
    if market_close is not None:
        stock_ret = stock_close.pct_change()
        market_ret = market_close.pct_change()
        common = stock_ret.index.intersection(market_ret.index)
        sr = stock_ret.reindex(common)
        mr = market_ret.reindex(common)

        # Rolling beta
        cov = sr.rolling(63, min_periods=21).cov(mr)
        var = mr.rolling(63, min_periods=21).var().clip(lower=1e-10)
        beta = cov / var

        # Relative strength (stock vs market)
        rel_strength_21 = stock_close.pct_change(21).reindex(common) - market_close.pct_change(21).reindex(common)
        rel_strength_63 = stock_close.pct_change(63).reindex(common) - market_close.pct_change(63).reindex(common)

        market_feats = pd.DataFrame({
            "market_beta": beta,
            "rel_strength_21d": rel_strength_21,
            "rel_strength_63d": rel_strength_63,
        }, index=common).reindex(stock_close.index)
        features.append(market_feats)

    # Combine all
    result = pd.concat(features, axis=1)

    # Handle duplicate columns
    result = result.loc[:, ~result.columns.duplicated()]

    # Drop rows with insufficient data
    core_cols = ["tam", "vol_21d", "atf_coiled_spring"]
    available_core = [c for c in core_cols if c in result.columns]
    if available_core:
        result = result.dropna(subset=available_core)

    return result
