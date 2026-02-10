"""
╔══════════════════════════════════════════════════════════════════════════╗
║  AthenaAI TRADING BOT v5.0 — BinaryOptionsToolsV2 / PocketOption     ║
║  Online-learning engine that improves with every trade it makes.       ║
║                                                                        ║
║  Features:                                                             ║
║   • 5-model ensemble (SGD, PA, NB + GBM, RandomForest)                ║
║   • 40+ engineered features from raw candle data                       ║
║   • AI EXPIRY SELECTION — picks optimal duration per trade             ║
║   • Automatic regime detection (trending / ranging / volatile)         ║
║   • Rolling performance tracker with adaptive cooldowns                ║
║   • Brain persistence — models survive restarts                        ║
║   • Full trade journal persisted to SQLite                             ║
║                                                                        ║
║  ⚠  USE ON DEMO FIRST.  Binary options carry extreme risk.             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pickle

# ---------------------------------------------------------------------------
# BinaryOptionsToolsV2 imports
# ---------------------------------------------------------------------------
try:
    from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
except ImportError:
    print("ERROR: BinaryOptionsToolsV2 not installed.")
    print("Install with:  pip install binaryoptionstoolsv2")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful fallback
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("WARNING: scikit-learn not found. Install with: pip install scikit-learn")
    print("The bot will run with a simplified rule-based fallback.\n")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("AIBot")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BotConfig:
    """All tuneable knobs in one place."""

    # --- connection ---
    ssid: str = ""                          # PocketOption session ID
    asset: str = "EURUSD"               # trading pair
    timeframe: int = 60                     # candle period (60s for data)

    # --- AI expiry selection ---
    expiry_options: tuple = (60, 120, 180, 300)  # seconds AI can pick from
    default_expiry: int = 60               # fallback / training label horizon

    # --- money management ---
    base_stake: float = 10.0                 # minimum trade size ($)
    max_stake: float = 100.0                 # hard ceiling ($)
    kelly_fraction: float = 0.50            # fraction of Kelly to use
    max_daily_loss: float = 300.0            # stop-loss for the day ($)
    max_concurrent_trades: int = 1          # max open trades

    # --- ML & signals ---
    warmup_candles: int = 60                 # candles before first trade
    min_confidence: float = 0.60            # lowered — batch models handle quality
    retrain_every: int = 10                 # partial_fit after N new samples
    lookback: int = 200                     # max candle history to keep
    feature_window: int = 20                # rolling window for features

    # --- signal readiness ---
    signal_confirmations: int = 1           # 1 = instant (no multi-candle wait)
    require_indicator_alignment: bool = False # disabled — ML already uses these
    skip_volatile_regime: bool = False       # let ML decide
    min_wait_between_trades: int = 60       # 1 min between trades

    # --- dataset pre-training ---
    dataset_path: str = ""                  # path to CSV for pre-training (optional)

    # --- risk / cooldown ---
    max_consec_losses: int = 3              # pause after 5 consecutive losses
    cooldown_seconds: int = 300             # 5 min cooldown
    regime_window: int = 30                 # candles for regime detection

    # --- persistence ---
    db_path: str = "trade_journal.db"
    brain_path: str = "athena_po_brain.pkl" # saved models

    # --- misc ---
    poll_interval: float = 2.0             # check every 2s


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════
class Direction(Enum):
    CALL = "call"
    PUT = "put"

class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

@dataclass
class TradeRecord:
    id: str
    direction: str
    asset: str
    stake: float
    confidence: float
    regime: str
    entry_time: float
    expiry: int = 120                      # AI-chosen expiry in seconds
    result: Optional[str] = None   # "win" / "loss" / "draw"
    profit: Optional[float] = None
    exit_time: Optional[float] = None
    features_json: Optional[str] = None  # stored feature vector as JSON


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (40 core + experimental features from raw OHLCV)
# ═══════════════════════════════════════════════════════════════════════════
class FeatureEngine:
    """Converts a window of candles into a numeric feature vector.
    Core features (always on) + experimental features (can be masked)."""

    # Names for logging — core features (indices 0-39)
    CORE_NAMES = [
        "body", "range", "mom_1", "mom_5",
        "ret_last", "ret_mean", "volatility",
        "price_sma5", "price_sma10", "price_sma20", "ma_cross", "macd_line",
        "macd_hist",
        "rsi", "rsi_zone",
        "bb_width", "bb_pos",
        "atr", "range_vs_atr",
        "stoch_k", "stoch_d", "stoch_diff",
        "cci", "williams_r", "adx",
        "rel_volume", "vol_std",
        "doji", "hammer", "engulfing",
        "ret_p25", "ret_p75", "skew", "kurt",
        "fractal_dim",
        "streak",
        "time_sin_h", "time_cos_h", "time_sin_d", "time_cos_d",
    ]

    # Experimental feature names (indices 40+)
    EXPERIMENTAL_NAMES = [
        "shooting_star", "bearish_engulfing",        # candlestick
        "sma50_dist", "sma50_above",                 # long trend
        "momentum_ratio",                             # 14/28 MA ratio
        "fib_dist_236", "fib_dist_382",              # fibonacci
        "fib_dist_500", "fib_dist_618",
        "support_dist", "resist_dist",               # S/R
        "hl_ratio", "oc_ratio",                      # price ratios
        "rsi_divergence", "volume_spike",            # divergence & spike
        "candle_wick_ratio", "body_vs_avg",          # candle anatomy
    ]

    NUM_CORE = len(CORE_NAMES)       # 40
    NUM_EXPERIMENTAL = len(EXPERIMENTAL_NAMES)  # 17

    def __init__(self):
        # Mask: 1.0 = active, 0.0 = masked off.  Core always 1.0
        self.feature_mask = np.ones(self.NUM_CORE + self.NUM_EXPERIMENTAL, dtype=np.float64)
        self.experimental_enabled = True  # master switch

    def compute(self, candles: list[Candle], window: int = 20) -> Optional[np.ndarray]:
        if len(candles) < max(window, 26):
            return None

        closes = np.array([c.close for c in candles])
        highs  = np.array([c.high  for c in candles])
        lows   = np.array([c.low   for c in candles])
        opens  = np.array([c.open  for c in candles])
        vols   = np.array([c.volume for c in candles])

        feats: list[float] = []

        # ============ CORE FEATURES (0-39) — always active ============

        # --- Price-action ---
        feats.append(closes[-1] - opens[-1])                     # current body
        feats.append(highs[-1] - lows[-1])                       # current range
        feats.append(closes[-1] - closes[-2])                    # 1-bar momentum
        feats.append(closes[-1] - closes[-5] if len(closes) >= 5 else 0)  # 5-bar mom

        # --- Returns ---
        rets = np.diff(closes) / (closes[:-1] + 1e-10)
        feats.append(rets[-1])                                   # latest return
        feats.append(np.mean(rets[-window:]))                    # mean return
        feats.append(np.std(rets[-window:]))                     # volatility

        # --- Moving averages ---
        sma5  = np.mean(closes[-5:])
        sma10 = np.mean(closes[-10:])
        sma20 = np.mean(closes[-window:])
        ema12 = FeatureEngine._ema(closes, 12)
        ema26 = FeatureEngine._ema(closes, 26)

        feats.append(closes[-1] - sma5)
        feats.append(closes[-1] - sma10)
        feats.append(closes[-1] - sma20)
        feats.append(sma5 - sma20)                               # MA cross
        feats.append(ema12 - ema26)                               # MACD line

        # --- MACD signal & histogram ---
        macd_line_arr = FeatureEngine._ema_array(closes, 12) - FeatureEngine._ema_array(closes, 26)
        signal_line = FeatureEngine._ema(macd_line_arr, 9)
        feats.append(macd_line_arr[-1] - signal_line)             # MACD histogram

        # --- RSI ---
        rsi = FeatureEngine._rsi(closes, 14)
        feats.append(rsi)
        feats.append(1.0 if rsi > 70 else (-1.0 if rsi < 30 else 0.0))  # overbought/sold

        # --- Bollinger Bands ---
        bb_mid = sma20
        bb_std = np.std(closes[-window:])
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        bb_pos = (closes[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        feats.append(bb_width)
        feats.append(bb_pos)

        # --- ATR ---
        atr = FeatureEngine._atr(highs, lows, closes, 14)
        feats.append(atr)
        feats.append((highs[-1] - lows[-1]) / (atr + 1e-10))    # current range vs ATR

        # --- Stochastic %K / %D ---
        stoch_k, stoch_d = FeatureEngine._stochastic(highs, lows, closes, 14, 3)
        feats.append(stoch_k)
        feats.append(stoch_d)
        feats.append(stoch_k - stoch_d)

        # --- CCI ---
        cci = FeatureEngine._cci(highs, lows, closes, 20)
        feats.append(cci)

        # --- Williams %R ---
        will_r = FeatureEngine._williams_r(highs, lows, closes, 14)
        feats.append(will_r)

        # --- ADX (simplified) ---
        adx = FeatureEngine._adx(highs, lows, closes, 14)
        feats.append(adx)

        # --- Volume features ---
        if np.any(vols > 0):
            feats.append(vols[-1] / (np.mean(vols[-window:]) + 1e-10))  # relative volume
            feats.append(np.std(vols[-window:]) / (np.mean(vols[-window:]) + 1e-10))
        else:
            feats.extend([1.0, 0.0])

        # --- Candle patterns (encoded) ---
        feats.append(FeatureEngine._doji(opens, highs, lows, closes))
        feats.append(FeatureEngine._hammer(opens, highs, lows, closes))
        feats.append(FeatureEngine._engulfing(opens, closes))

        # --- Higher-order stats ---
        feats.append(float(np.percentile(rets[-window:], 25)))
        feats.append(float(np.percentile(rets[-window:], 75)))
        skew = FeatureEngine._skewness(rets[-window:])
        kurt = FeatureEngine._kurtosis(rets[-window:])
        feats.append(skew)
        feats.append(kurt)

        # --- Fractal dimension (Higuchi approx) ---
        feats.append(FeatureEngine._higuchi_fd(closes[-window:]))

        # --- Streak features ---
        streak = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i - 1]:
                streak += 1
            elif closes[i] < closes[i - 1]:
                streak -= 1
            else:
                break
            if abs(streak) >= 10:
                break
        feats.append(float(streak))

        # --- Time features (cyclical) ---
        ts = candles[-1].timestamp
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        feats.append(math.sin(2 * math.pi * hour / 24))
        feats.append(math.cos(2 * math.pi * hour / 24))
        dow = dt.weekday()
        feats.append(math.sin(2 * math.pi * dow / 7))
        feats.append(math.cos(2 * math.pi * dow / 7))

        # ============ EXPERIMENTAL FEATURES (40+) — can be masked ============
        if self.experimental_enabled:
            # Shooting star pattern
            body = abs(closes[-1] - opens[-1])
            upper_wick = highs[-1] - max(opens[-1], closes[-1])
            lower_wick = min(opens[-1], closes[-1]) - lows[-1]
            feats.append(1.0 if (body > 1e-10 and upper_wick > 2 * body and lower_wick < body) else 0.0)

            # Bearish engulfing
            if len(opens) >= 2:
                prev_bull = closes[-2] > opens[-2]
                curr_bear = closes[-1] < opens[-1]
                engulfs = opens[-1] > closes[-2] and closes[-1] < opens[-2]
                feats.append(1.0 if (prev_bull and curr_bear and engulfs) else 0.0)
            else:
                feats.append(0.0)

            # SMA 50
            if len(closes) >= 50:
                sma50 = np.mean(closes[-50:])
                feats.append(closes[-1] - sma50)
                feats.append(1.0 if closes[-1] > sma50 else -1.0)
            else:
                feats.extend([0.0, 0.0])

            # Price momentum ratio (14/28 MA)
            if len(closes) >= 28:
                feats.append(np.mean(closes[-14:]) / (np.mean(closes[-28:]) + 1e-10))
            else:
                feats.append(1.0)

            # Fibonacci distances
            if len(highs) >= 50:
                recent_high = np.max(highs[-50:])
                recent_low = np.min(lows[-50:])
                fib_range = recent_high - recent_low
                price = closes[-1]
                feats.append((price - (recent_high - 0.236 * fib_range)) / (fib_range + 1e-10))
                feats.append((price - (recent_high - 0.382 * fib_range)) / (fib_range + 1e-10))
                feats.append((price - (recent_high - 0.500 * fib_range)) / (fib_range + 1e-10))
                feats.append((price - (recent_high - 0.618 * fib_range)) / (fib_range + 1e-10))
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])

            # Support / Resistance distance
            support = np.min(lows[-window:])
            resistance = np.max(highs[-window:])
            price = closes[-1]
            feats.append((price - support) / (support + 1e-10))
            feats.append((resistance - price) / (price + 1e-10))

            # High/Low ratio & Open/Close ratio
            feats.append(highs[-1] / (lows[-1] + 1e-10))
            feats.append(opens[-1] / (closes[-1] + 1e-10))

            # RSI divergence (price making higher highs but RSI making lower highs)
            if len(closes) >= 28:
                rsi_arr = []
                for j in range(max(0, len(closes) - 14), len(closes)):
                    rsi_arr.append(FeatureEngine._rsi(closes[:j+1], 14))
                price_trend = closes[-1] - closes[-14]
                rsi_trend = rsi_arr[-1] - rsi_arr[0] if len(rsi_arr) >= 2 else 0
                feats.append(1.0 if (price_trend > 0 and rsi_trend < -5) else
                             (-1.0 if (price_trend < 0 and rsi_trend > 5) else 0.0))
            else:
                feats.append(0.0)

            # Volume spike (current volume vs 20-period average)
            if np.any(vols > 0):
                avg_vol = np.mean(vols[-window:])
                feats.append(vols[-1] / (avg_vol + 1e-10) if avg_vol > 0 else 1.0)
            else:
                feats.append(1.0)

            # Candle wick ratio (total wicks / body)
            body = abs(closes[-1] - opens[-1])
            total_wick = (highs[-1] - lows[-1]) - body
            feats.append(total_wick / (body + 1e-10))

            # Body vs average body (is this candle unusually big/small?)
            avg_body = np.mean(np.abs(closes[-window:] - opens[-window:]))
            feats.append(body / (avg_body + 1e-10))

        else:
            # Experimental disabled — fill with zeros to keep dimensions
            feats.extend([0.0] * self.NUM_EXPERIMENTAL)

        # ============ APPLY MASK ============
        result = np.array(feats, dtype=np.float64)
        if len(result) == len(self.feature_mask):
            result *= self.feature_mask
        return result

    # ---- Helpers ----
    @staticmethod
    def _ema(data: np.ndarray, span: int) -> float:
        if len(data) < span:
            return float(np.mean(data))
        alpha = 2.0 / (span + 1)
        val = float(data[0])
        for d in data[1:]:
            val = alpha * d + (1 - alpha) * val
        return val

    @staticmethod
    def _ema_array(data: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        out = np.empty_like(data, dtype=np.float64)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gain = np.mean(np.maximum(deltas, 0))
        loss = np.mean(np.maximum(-deltas, 0))
        if loss < 1e-10:
            return 100.0
        rs = gain / loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _atr(highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return float(np.mean(highs[-period:] - lows[-period:]))
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _stochastic(highs, lows, closes, k_period=14, d_period=3):
        if len(closes) < k_period:
            return 50.0, 50.0
        low_min = np.min(lows[-k_period:])
        high_max = np.max(highs[-k_period:])
        k = 100.0 * (closes[-1] - low_min) / (high_max - low_min + 1e-10)
        # Simple %D
        k_vals = []
        for i in range(min(d_period, len(closes) - k_period + 1)):
            idx = -(1 + i)
            lm = np.min(lows[idx - k_period + 1: len(lows) + idx + 1]) if idx != -1 else low_min
            hm = np.max(highs[idx - k_period + 1: len(highs) + idx + 1]) if idx != -1 else high_max
            k_vals.append(100.0 * (closes[idx] - lm) / (hm - lm + 1e-10))
        d = float(np.mean(k_vals)) if k_vals else k
        return k, d

    @staticmethod
    def _cci(highs, lows, closes, period=20):
        if len(closes) < period:
            return 0.0
        tp = (highs[-period:] + lows[-period:] + closes[-period:]) / 3.0
        sma = np.mean(tp)
        mad = np.mean(np.abs(tp - sma))
        return (tp[-1] - sma) / (0.015 * mad + 1e-10)

    @staticmethod
    def _williams_r(highs, lows, closes, period=14):
        if len(closes) < period:
            return -50.0
        hh = np.max(highs[-period:])
        ll = np.min(lows[-period:])
        return -100.0 * (hh - closes[-1]) / (hh - ll + 1e-10)

    @staticmethod
    def _adx(highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return 25.0
        up = highs[1:] - highs[:-1]
        down = lows[:-1] - lows[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr = FeatureEngine._atr(highs, lows, closes, period)
        plus_di = 100 * np.mean(plus_dm[-period:]) / (atr + 1e-10)
        minus_di = 100 * np.mean(minus_dm[-period:]) / (atr + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx

    @staticmethod
    def _doji(opens, highs, lows, closes):
        body = abs(closes[-1] - opens[-1])
        rng = highs[-1] - lows[-1]
        return 1.0 if rng > 0 and body / rng < 0.1 else 0.0

    @staticmethod
    def _hammer(opens, highs, lows, closes):
        body = abs(closes[-1] - opens[-1])
        lower_wick = min(opens[-1], closes[-1]) - lows[-1]
        upper_wick = highs[-1] - max(opens[-1], closes[-1])
        rng = highs[-1] - lows[-1]
        if rng < 1e-10:
            return 0.0
        if lower_wick > 2 * body and upper_wick < body:
            return 1.0
        if upper_wick > 2 * body and lower_wick < body:
            return -1.0
        return 0.0

    @staticmethod
    def _engulfing(opens, closes):
        if len(opens) < 2:
            return 0.0
        prev_body = closes[-2] - opens[-2]
        curr_body = closes[-1] - opens[-1]
        if prev_body < 0 and curr_body > 0 and curr_body > abs(prev_body):
            return 1.0   # bullish engulfing
        if prev_body > 0 and curr_body < 0 and abs(curr_body) > prev_body:
            return -1.0  # bearish engulfing
        return 0.0

    @staticmethod
    def _skewness(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-10:
            return 0.0
        return float(np.mean(((arr - m) / s) ** 3))

    @staticmethod
    def _kurtosis(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-10:
            return 0.0
        return float(np.mean(((arr - m) / s) ** 4) - 3.0)

    @staticmethod
    def _higuchi_fd(series, kmax=5):
        n = len(series)
        if n < kmax * 2:
            return 1.5
        lk = []
        for k in range(1, kmax + 1):
            lengths = []
            for m in range(1, k + 1):
                idxs = np.arange(m - 1, n, k)
                if len(idxs) < 2:
                    continue
                vals = series[idxs]
                length = np.sum(np.abs(np.diff(vals))) * (n - 1) / (k * len(idxs) * k)
                lengths.append(length)
            if lengths:
                lk.append(np.mean(lengths))
        if len(lk) < 2 or any(l <= 0 for l in lk):
            return 1.5
        log_k = np.log(np.arange(1, len(lk) + 1, dtype=np.float64))
        log_l = np.log(np.array(lk))
        slope = np.polyfit(log_k, log_l, 1)[0]
        return float(-slope)


# ═══════════════════════════════════════════════════════════════════════════
#  ONLINE LEARNING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════
class EnsemblePredictor:
    """
    5-model ensemble: 3 online learners + 2 batch powerhouses.
    Online:  SGD, Passive-Aggressive, Naive Bayes (adapt in real-time)
    Batch:   GradientBoosting, RandomForest (trained once, high accuracy)
    Batch models get 2× vote weight since they're stronger.
    """

    def __init__(self):
        if not SKLEARN_OK:
            self.models = []
            self.scaler = None
            return

        self.scaler = StandardScaler()

        # Online models — support partial_fit for live learning
        self.models = [
            {
                "name": "SGD",
                "clf": SGDClassifier(
                    loss="modified_huber", penalty="l2",
                    alpha=1e-4, warm_start=True, random_state=42,
                ),
                "accuracy_ema": 0.5,
            },
            {
                "name": "PA",
                "clf": SGDClassifier(
                    loss="hinge", penalty=None,
                    learning_rate="pa1", eta0=1.0,
                    warm_start=True, random_state=42,
                ),
                "accuracy_ema": 0.5,
            },
            {
                "name": "NB",
                "clf": GaussianNB(),
                "accuracy_ema": 0.5,
            },
        ]

        # Batch models — much stronger, trained once on full dataset
        self._batch_models = [
            {
                "name": "GBM",
                "clf": GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                ),
                "accuracy_ema": 0.5,
            },
            {
                "name": "RF",
                "clf": RandomForestClassifier(
                    n_estimators=200, max_depth=8,
                    random_state=42, n_jobs=-1,
                ),
                "accuracy_ema": 0.5,
            },
        ]
        self._batch_fitted = False
        self._batch_X: list[np.ndarray] = []
        self._batch_y: list[int] = []

        self._fitted = False
        self._X_buffer: list[np.ndarray] = []
        self._y_buffer: list[int] = []       # 1 = CALL-win, 0 = PUT-win
        self._classes = np.array([0, 1])

    # -- incremental training --
    def add_sample(self, features: np.ndarray, label: int):
        self._X_buffer.append(features)
        self._y_buffer.append(label)
        self._batch_X.append(features)
        self._batch_y.append(label)

    def partial_fit(self):
        """Train online models on buffered samples, then clear."""
        if not SKLEARN_OK or len(self._X_buffer) == 0:
            return

        X = np.vstack(self._X_buffer)
        y = np.array(self._y_buffer)

        # Reset if feature dimension changed
        if self._fitted and hasattr(self.scaler, 'n_features_in_'):
            if self.scaler.n_features_in_ != X.shape[1]:
                log.warning("Feature dimension changed (%d → %d) — resetting.",
                            self.scaler.n_features_in_, X.shape[1])
                self.__init__()
                return

        if not self._fitted:
            self.scaler.fit(X)
        else:
            self.scaler.partial_fit(X)

        X_scaled = self.scaler.transform(X)

        for m in self.models:
            clf = m["clf"]
            if hasattr(clf, "partial_fit"):
                clf.partial_fit(X_scaled, y, classes=self._classes)
            else:
                clf.fit(X_scaled, y)

            if self._fitted:
                preds = clf.predict(X_scaled)
                acc = float(np.mean(preds == y))
                m["accuracy_ema"] = 0.9 * m["accuracy_ema"] + 0.1 * acc

        self._fitted = True
        self._X_buffer.clear()
        self._y_buffer.clear()
        log.info(
            "Models updated  |  acc EMAs: %s",
            {m["name"]: f'{m["accuracy_ema"]:.3f}' for m in self.models},
        )

    def train_batch_models(self):
        """Train GBM + RF on ALL accumulated data. Call after dataset load."""
        if len(self._batch_X) < 100:
            log.warning("Not enough data for batch models (%d)", len(self._batch_X))
            return
        X = np.vstack(self._batch_X)
        y = np.array(self._batch_y)
        X_scaled = self.scaler.transform(X)

        log.info("🧠 Training batch models (GBM + RF) on %d samples …", len(X))
        for m in self._batch_models:
            try:
                m["clf"].fit(X_scaled, y)
                preds = m["clf"].predict(X_scaled)
                acc = float(np.mean(preds == y))
                m["accuracy_ema"] = acc
                log.info("   %s trained — accuracy: %.1f%%", m["name"], acc * 100)
            except Exception as e:
                log.warning("   %s failed: %s", m["name"], e)
        self._batch_fitted = True

    # -- persistence --
    def save_brain(self, path="athena_po_brain.pkl"):
        if not self._fitted:
            log.warning("No trained models to save.")
            return
        state = {
            "scaler": self.scaler,
            "models": [(m["name"], m["clf"], m["accuracy_ema"]) for m in self.models],
            "batch_models": [(m["name"], m["clf"], m["accuracy_ema"]) for m in self._batch_models],
            "batch_fitted": self._batch_fitted,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info("🧠 Brain saved to %s", path)

    def load_brain(self, path="athena_po_brain.pkl") -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.scaler = state["scaler"]
            for saved, m in zip(state.get("models", []), self.models):
                m["name"], m["clf"], m["accuracy_ema"] = saved[0], saved[1], saved[2]
            if "batch_models" in state:
                for saved, m in zip(state["batch_models"], self._batch_models):
                    m["name"], m["clf"], m["accuracy_ema"] = saved[0], saved[1], saved[2]
                self._batch_fitted = state.get("batch_fitted", False)
            self._fitted = state["fitted"]
            log.info("🧠 Brain loaded (fitted=%s, batch=%s, models: %s)",
                     self._fitted, self._batch_fitted,
                     {m["name"]: f'{m["accuracy_ema"]:.3f}' for m in self.models})
            return True
        except Exception as e:
            log.warning("Failed to load brain: %s", e)
            return False

    # -- prediction --
    def predict(self, features: np.ndarray) -> tuple[Direction, float]:
        if not SKLEARN_OK or not self._fitted:
            return self._fallback_predict(features)

        try:
            if not hasattr(self.scaler, 'n_features_in_'):
                return self._fallback_predict(features)
            if self.scaler.n_features_in_ != len(features):
                return self._fallback_predict(features)
        except Exception:
            return self._fallback_predict(features)

        try:
            X = self.scaler.transform(features.reshape(1, -1))
        except Exception:
            return self._fallback_predict(features)

        weighted_call = 0.0
        total_weight = 0.0

        # Online models vote
        for m in self.models:
            w = m["accuracy_ema"]
            clf = m["clf"]
            if hasattr(clf, "predict_proba"):
                try:
                    proba = clf.predict_proba(X)[0]
                    p_call = proba[1] if len(proba) > 1 else 0.5
                except Exception:
                    p_call = 0.5
            else:
                pred = clf.predict(X)[0]
                p_call = 1.0 if pred == 1 else 0.0

            weighted_call += w * p_call
            total_weight += w

        # Batch models vote (2× weight — they're stronger)
        if self._batch_fitted:
            for m in self._batch_models:
                w = m["accuracy_ema"] * 2.0
                try:
                    if hasattr(m["clf"], "predict_proba"):
                        proba = m["clf"].predict_proba(X)[0]
                        p_call = proba[1] if len(proba) > 1 else 0.5
                    else:
                        p_call = 1.0 if m["clf"].predict(X)[0] == 1 else 0.0
                except Exception:
                    p_call = 0.5
                weighted_call += w * p_call
                total_weight += w

        p = weighted_call / (total_weight + 1e-10)

        if p >= 0.5:
            return Direction.CALL, p
        else:
            return Direction.PUT, 1.0 - p

    @staticmethod
    def _fallback_predict(features: np.ndarray) -> tuple[Direction, float]:
        """Rule-based fallback when sklearn is missing or models untrained."""
        rsi = features[14] if len(features) > 14 else 50.0
        macd_hist = features[12] if len(features) > 12 else 0.0
        sma_cross = features[10] if len(features) > 10 else 0.0

        score = 0.0
        if rsi < 30:
            score += 0.3
        elif rsi > 70:
            score -= 0.3
        if macd_hist > 0:
            score += 0.2
        elif macd_hist < 0:
            score -= 0.2
        if sma_cross > 0:
            score += 0.15
        elif sma_cross < 0:
            score -= 0.15

        conf = 0.5 + min(abs(score), 0.45)
        if score > 0:
            return Direction.CALL, conf
        else:
            return Direction.PUT, conf


# ═══════════════════════════════════════════════════════════════════════════
#  REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
class RegimeDetector:
    @staticmethod
    def detect(candles: list[Candle], window: int = 30) -> Regime:
        if len(candles) < window:
            return Regime.RANGING

        closes = np.array([c.close for c in candles[-window:]])
        rets = np.diff(closes) / (closes[:-1] + 1e-10)

        trend = np.polyfit(np.arange(len(closes)), closes, 1)[0]
        vol = np.std(rets)
        mean_ret = np.mean(rets)

        # Normalise trend by price level
        rel_trend = trend / (closes[-1] + 1e-10) * window

        if vol > 0.005:
            return Regime.VOLATILE
        if rel_trend > 0.002:
            return Regime.TRENDING_UP
        if rel_trend < -0.002:
            return Regime.TRENDING_DOWN
        return Regime.RANGING


# ═══════════════════════════════════════════════════════════════════════════
#  EXPIRY SELECTOR — AI picks optimal trade duration per-trade
# ═══════════════════════════════════════════════════════════════════════════
class ExpirySelector:
    """
    Chooses the best expiry for each trade based on:
      • Regime  → trending = longer, volatile/ranging = shorter
      • ATR / volatility → high vol = shorter exposure
      • ADX (trend strength) → strong trend = ride it longer
      • Confidence → high confidence = can go longer
      • RSI extremes → overbought/oversold = shorter (reversal)
      • Past performance → learns which durations actually win

    PocketOption expiry options: 60s, 120s, 180s, 300s (configurable)
    """

    def __init__(self, expiry_options: tuple = (60, 120, 180, 300)):
        self.options = sorted(expiry_options)
        # Track win/loss per expiry duration — this learns over time
        self.stats: dict[int, dict] = {e: {"wins": 0, "losses": 0} for e in self.options}

    def select(self, regime: Regime, features: np.ndarray, confidence: float) -> int:
        """Pick the best expiry given current market conditions.
        Returns expiry in seconds."""

        # Extract key indicators from feature vector (see FeatureEngine.CORE_NAMES)
        volatility = features[6] if len(features) > 6 else 0.01   # idx 6
        rsi        = features[13] if len(features) > 13 else 50.0  # idx 13
        atr        = features[17] if len(features) > 17 else 0.001 # idx 17
        adx        = features[24] if len(features) > 24 else 25.0  # idx 24

        scores: dict[int, float] = {}

        for exp in self.options:
            score = 0.0

            # ── 1. Regime ──
            if regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
                # Trending → prefer longer (ride the wave)
                if exp >= 300:   score += 3.0
                elif exp >= 180: score += 2.0
                elif exp >= 120: score += 1.0
                else:            score += 0.5
            elif regime == Regime.VOLATILE:
                # Volatile → prefer shorter (less exposure to reversals)
                if exp <= 60:    score += 3.0
                elif exp <= 120: score += 2.0
                elif exp <= 180: score += 1.0
                else:            score -= 1.0
            else:
                # Ranging → short to medium
                if exp <= 120:   score += 2.5
                elif exp <= 180: score += 2.0
                elif exp <= 300: score += 1.0
                else:            score += 0.5

            # ── 2. Volatility (return std) ──
            if volatility > 0.003:
                # High vol → shorter is safer
                if exp <= 120: score += 1.5
                if exp >= 300: score -= 1.0
            elif volatility < 0.001:
                # Low vol → need longer to see movement
                if exp >= 180: score += 1.0
                if exp <= 60:  score -= 0.5

            # ── 3. ADX (trend strength) ──
            if adx > 30:
                # Strong trend → go longer
                if exp >= 180: score += 1.5
                if exp >= 300: score += 0.5
            elif adx < 15:
                # No trend → stay short
                if exp <= 120: score += 1.0

            # ── 4. Confidence ──
            if confidence >= 0.75:
                # Very confident → can afford longer expiry
                if exp >= 180: score += 1.5
            elif confidence >= 0.65:
                if exp >= 120: score += 0.5
            else:
                # Lower confidence → shorter = less risk
                if exp <= 120: score += 1.0
                if exp >= 300: score -= 1.0

            # ── 5. RSI extremes → reversal likely → shorter ──
            if rsi > 75 or rsi < 25:
                if exp <= 120: score += 1.0
                if exp >= 300: score -= 0.5

            # ── 6. Past performance (adaptive learning) ──
            st = self.stats.get(exp, {"wins": 0, "losses": 0})
            total = st["wins"] + st["losses"]
            if total >= 5:
                wr = st["wins"] / total
                # Boost winners, penalise losers: 60% WR → +0.5, 40% → -0.5
                score += (wr - 0.50) * 5.0

            scores[exp] = score

        best = max(scores, key=scores.get)
        return best

    def record_result(self, expiry: int, result: str):
        """Feed trade result back — learn which expiries work."""
        if expiry not in self.stats:
            self.stats[expiry] = {"wins": 0, "losses": 0}
        if result == "win":
            self.stats[expiry]["wins"] += 1
        elif result == "loss":
            self.stats[expiry]["losses"] += 1

    def status_line(self) -> str:
        parts = []
        for exp in self.options:
            st = self.stats.get(exp, {"wins": 0, "losses": 0})
            total = st["wins"] + st["losses"]
            if total > 0:
                wr = st["wins"] / total * 100
                parts.append(f"{exp}s:{wr:.0f}%({total})")
        return " | ".join(parts) if parts else "no data yet"

    def save_state(self) -> dict:
        return {"stats": dict(self.stats), "options": list(self.options)}

    def load_state(self, state: dict):
        if "stats" in state:
            for k, v in state["stats"].items():
                self.stats[int(k)] = v


# ═══════════════════════════════════════════════════════════════════════════
#  MONEY MANAGEMENT (Kelly-fractional)
# ═══════════════════════════════════════════════════════════════════════════
class MoneyManager:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.daily_pnl = 0.0
        self.day_start = datetime.now(timezone.utc).date()

    def reset_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day_start:
            log.info("New day — resetting daily P&L tracker")
            self.daily_pnl = 0.0
            self.day_start = today

    def can_trade(self) -> bool:
        self.reset_if_new_day()
        return self.daily_pnl > -self.cfg.max_daily_loss

    def compute_stake(self, confidence: float, win_rate: float, payout: float = 0.85) -> float:
        """Kelly criterion capped by config limits."""
        if payout <= 0:
            return self.cfg.base_stake
        # Kelly: f* = (p*b - q) / b  where b=payout, p=win_rate, q=1-p
        edge = win_rate * payout - (1 - win_rate)
        if edge <= 0:
            return self.cfg.base_stake
        kelly = edge / payout
        fraction = kelly * self.cfg.kelly_fraction
        stake = self.cfg.base_stake + fraction * (self.cfg.max_stake - self.cfg.base_stake)
        # Scale by confidence
        stake *= (confidence - 0.5) * 2  # maps [0.5, 1.0] → [0, 1]
        stake = max(self.cfg.base_stake, min(stake, self.cfg.max_stake))
        return round(stake, 2)

    def record(self, pnl: float):
        self.daily_pnl += pnl


# ═══════════════════════════════════════════════════════════════════════════
#  TRADE JOURNAL (SQLite)
# ═══════════════════════════════════════════════════════════════════════════
class TradeJournal:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          TEXT PRIMARY KEY,
                direction   TEXT,
                asset       TEXT,
                stake       REAL,
                confidence  REAL,
                regime      TEXT,
                entry_time  REAL,
                exit_time   REAL,
                result      TEXT,
                profit      REAL,
                features    TEXT,
                expiry      INTEGER
            )
        """)
        # Migrate old DB: add columns if missing
        for col, ctype in [("features", "TEXT"), ("expiry", "INTEGER")]:
            try:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {ctype}")
            except Exception:
                pass  # column already exists
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_snapshots (
                ts          REAL,
                win_rate    REAL,
                total_trades INTEGER,
                daily_pnl   REAL,
                regime      TEXT
            )
        """)
        self.conn.commit()

    def save_trade(self, t: TradeRecord):
        self.conn.execute(
            "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.id, t.direction, t.asset, t.stake, t.confidence,
             t.regime, t.entry_time, t.exit_time, t.result, t.profit,
             t.features_json, t.expiry),
        )
        self.conn.commit()

    def load_completed_trades(self) -> list[dict]:
        """Load all completed trades with features for retraining."""
        cur = self.conn.execute(
            "SELECT direction, result, features FROM trades "
            "WHERE result IN ('win', 'loss') AND features IS NOT NULL "
            "ORDER BY entry_time ASC"
        )
        rows = cur.fetchall()
        trades = []
        for direction, result, features_json in rows:
            trades.append({
                "direction": direction,
                "result": result,
                "features_json": features_json,
            })
        return trades

    def save_snapshot(self, win_rate, total, daily_pnl, regime):
        self.conn.execute(
            "INSERT INTO model_snapshots VALUES (?,?,?,?,?)",
            (time.time(), win_rate, total, daily_pnl, regime),
        )
        self.conn.commit()

    def recent_win_rate(self, n: int = 50) -> float:
        cur = self.conn.execute(
            "SELECT result FROM trades WHERE result IS NOT NULL ORDER BY entry_time DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
        if not rows:
            return 0.5
        wins = sum(1 for r in rows if r[0] == "win")
        return wins / len(rows)

    def total_trades(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM trades")
        return cur.fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════════
#  PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════
class PerformanceTracker:
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_profit = 0.0
        self.consec_losses = 0
        self.max_drawdown = 0.0
        self._peak = 0.0
        self.recent_results: deque[str] = deque(maxlen=100)

    @property
    def total(self):
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self):
        t = self.wins + self.losses
        return self.wins / t if t > 0 else 0.5

    def record(self, result: str, profit: float):
        self.recent_results.append(result)
        self.total_profit += profit
        if result == "win":
            self.wins += 1
            self.consec_losses = 0
        elif result == "loss":
            self.losses += 1
            self.consec_losses += 1
        else:
            self.draws += 1

        # Drawdown
        if self.total_profit > self._peak:
            self._peak = self.total_profit
        dd = self._peak - self.total_profit
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def summary(self) -> str:
        return (
            f"W:{self.wins} L:{self.losses} D:{self.draws} "
            f"WR:{self.win_rate:.1%} "
            f"P&L:${self.total_profit:+.2f} "
            f"MaxDD:${self.max_drawdown:.2f} "
            f"Streak:{'L' if self.consec_losses else 'OK'}{self.consec_losses}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE LAB — tracks feature importance, enables/disables experiments
# ═══════════════════════════════════════════════════════════════════════════
class FeatureLab:
    """
    Tracks how correlated each feature is with correct predictions.
    Periodically reviews and masks anti-predictive features.
    """

    def __init__(self, feature_engine: FeatureEngine, review_interval: int = 50):
        self.engine = feature_engine
        self.review_interval = review_interval
        self._trade_count = 0

        total = FeatureEngine.NUM_CORE + FeatureEngine.NUM_EXPERIMENTAL
        self.all_names = FeatureEngine.CORE_NAMES + FeatureEngine.EXPERIMENTAL_NAMES

        # Running stats: for each feature, track correlation with win/loss
        self._win_sums = np.zeros(total, dtype=np.float64)
        self._loss_sums = np.zeros(total, dtype=np.float64)
        self._win_count = 0
        self._loss_count = 0

    def record_trade(self, features: np.ndarray, result: str):
        """Feed a trade's features and outcome."""
        if result == "win":
            self._win_sums += features[:len(self._win_sums)]
            self._win_count += 1
        elif result == "loss":
            self._loss_sums += features[:len(self._loss_sums)]
            self._loss_count += 1

        self._trade_count += 1
        if self._trade_count % self.review_interval == 0:
            self._review()

    def _review(self):
        """Analyze feature importance and adjust masks."""
        if self._win_count < 20 or self._loss_count < 20:
            return  # not enough data

        log.info("🔬 Feature Lab Review (after %d trades):", self._trade_count)

        # Average feature values for wins vs losses
        win_avg = self._win_sums / self._win_count
        loss_avg = self._loss_sums / self._loss_count

        # Importance = |win_avg - loss_avg| / (std + epsilon)
        # Features that differ most between wins and losses are most informative
        combined_avg = (self._win_sums + self._loss_sums) / (self._win_count + self._loss_count)
        diff = np.abs(win_avg - loss_avg)

        # Only evaluate experimental features for masking
        num_core = FeatureEngine.NUM_CORE

        # --- Report top 5 most predictive features ---
        importance = diff.copy()
        top_indices = np.argsort(importance)[::-1][:5]
        log.info("   🏆 Top 5 most predictive features:")
        for idx in top_indices:
            name = self.all_names[idx] if idx < len(self.all_names) else f"feat_{idx}"
            direction = "↑WIN" if win_avg[idx] > loss_avg[idx] else "↑LOSS"
            log.info("      %s: importance=%.4f (%s)", name, importance[idx], direction)

        # --- Report bottom 5 least predictive ---
        bottom_indices = np.argsort(importance)[:5]
        log.info("   📉 Bottom 5 least predictive features:")
        for idx in bottom_indices:
            name = self.all_names[idx] if idx < len(self.all_names) else f"feat_{idx}"
            log.info("      %s: importance=%.6f", name, importance[idx])

        # --- Mask anti-predictive experimental features ---
        masked_count = 0
        unmasked_count = 0
        for i in range(num_core, len(importance)):
            if importance[i] < 1e-6:
                # Feature shows zero difference between wins/losses — mask it
                self.engine.feature_mask[i] = 0.0
                masked_count += 1
            else:
                # Feature shows some signal — keep it active
                self.engine.feature_mask[i] = 1.0
                unmasked_count += 1

        active_experimental = int(np.sum(self.engine.feature_mask[num_core:]))
        log.info("   🧪 Experimental features: %d active, %d masked",
                 active_experimental, masked_count)

    def get_report(self) -> str:
        """Short status string."""
        num_core = FeatureEngine.NUM_CORE
        active = int(np.sum(self.engine.feature_mask[num_core:]))
        total = FeatureEngine.NUM_EXPERIMENTAL
        return f"features: {num_core}+{active}/{total}exp"


# ═══════════════════════════════════════════════════════════════════════════
#  ADAPTIVE STRATEGY — self-tuning meta-layer
# ═══════════════════════════════════════════════════════════════════════════
class AdaptiveStrategy:
    """
    Tracks win/loss rates across multiple dimensions (regime, hour, direction,
    confidence band) and dynamically adjusts trading parameters.
    Reviewed every `review_interval` trades.
    """

    def __init__(self, review_interval: int = 25, min_samples: int = 15):
        self.review_interval = review_interval
        self.min_samples = min_samples  # need this many trades before making decisions
        self._trade_count = 0

        # --- per-dimension tracking ---
        # Each stores {key: {"wins": int, "losses": int}}
        self.by_regime: dict[str, dict] = {}
        self.by_hour: dict[int, dict] = {}          # 0-23 UTC hour
        self.by_direction: dict[str, dict] = {}     # "call" / "put"
        self.by_conf_band: dict[str, dict] = {}     # "low"/"med"/"high"

        # --- adaptive outputs ---
        self.blocked_regimes: set[str] = set()       # regimes to avoid
        self.blocked_hours: set[int] = set()          # hours to avoid
        self.confidence_adj: float = 0.0              # added to min_confidence
        self.preferred_direction: Optional[str] = None  # None = both OK
        self.adaptive_cooldown: int = 0               # extra seconds after loss streak

        # --- rolling recent window (last 50 trades) ---
        self._recent: deque[dict] = deque(maxlen=50)

    def _bucket(self) -> dict:
        return {"wins": 0, "losses": 0}

    def _wr(self, bucket: dict) -> float:
        total = bucket["wins"] + bucket["losses"]
        return bucket["wins"] / total if total > 0 else 0.5

    def _conf_band(self, confidence: float) -> str:
        if confidence < 0.70:
            return "low"
        elif confidence < 0.80:
            return "med"
        else:
            return "high"

    # ------------------------------------------------------------------
    def record_trade(self, direction: str, regime: str, confidence: float,
                     hour: int, result: str):
        """Feed trade outcome into the tracker."""
        self._trade_count += 1
        outcome = "wins" if result == "win" else "losses"

        # Store in all dimensions
        for store, key in [
            (self.by_regime, regime),
            (self.by_hour, hour),
            (self.by_direction, direction),
            (self.by_conf_band, self._conf_band(confidence)),
        ]:
            if key not in store:
                store[key] = self._bucket()
            store[key][outcome] += 1

        # Rolling recent
        self._recent.append({
            "direction": direction, "regime": regime,
            "confidence": confidence, "hour": hour, "result": result,
        })

        # Run review periodically
        if self._trade_count % self.review_interval == 0:
            self._review()

    # ------------------------------------------------------------------
    def _review(self):
        """Analyze all dimensions and adjust strategy parameters."""
        log.info("🧠 Adaptive Strategy Review (after %d trades):", self._trade_count)

        # --- 1. Regime analysis: block regimes with bad win rates ---
        self.blocked_regimes.clear()
        for regime, stats in self.by_regime.items():
            total = stats["wins"] + stats["losses"]
            if total >= self.min_samples:
                wr = self._wr(stats)
                if wr < 0.48:  # losing regime
                    self.blocked_regimes.add(regime)
                    log.info("   ⛔ Blocking regime '%s' (WR: %.1f%% over %d trades)",
                             regime, wr * 100, total)
                else:
                    log.info("   ✅ Regime '%s': %.1f%% WR (%d trades)",
                             regime, wr * 100, total)

        # --- 2. Hour analysis: block consistently bad hours ---
        self.blocked_hours.clear()
        for hour, stats in sorted(self.by_hour.items()):
            total = stats["wins"] + stats["losses"]
            if total >= self.min_samples:
                wr = self._wr(stats)
                if wr < 0.47:  # bad hour
                    self.blocked_hours.add(hour)
                    log.info("   ⛔ Blocking hour %02d:00 UTC (WR: %.1f%% over %d trades)",
                             hour, wr * 100, total)

        # --- 3. Direction analysis ---
        self.preferred_direction = None
        for direction, stats in self.by_direction.items():
            total = stats["wins"] + stats["losses"]
            if total >= self.min_samples:
                wr = self._wr(stats)
                log.info("   📊 Direction '%s': %.1f%% WR (%d trades)",
                         direction, wr * 100, total)

        # If one direction is clearly bad, bias away from it
        call_stats = self.by_direction.get("call", self._bucket())
        put_stats = self.by_direction.get("put", self._bucket())
        call_total = call_stats["wins"] + call_stats["losses"]
        put_total = put_stats["wins"] + put_stats["losses"]

        if call_total >= self.min_samples and put_total >= self.min_samples:
            call_wr = self._wr(call_stats)
            put_wr = self._wr(put_stats)
            # Only block a direction if it's clearly losing AND the other is winning
            if call_wr < 0.45 and put_wr > 0.55:
                self.preferred_direction = "put"
                log.info("   🔄 Favoring PUT trades (CALL WR too low: %.1f%%)", call_wr * 100)
            elif put_wr < 0.45 and call_wr > 0.55:
                self.preferred_direction = "call"
                log.info("   🔄 Favoring CALL trades (PUT WR too low: %.1f%%)", put_wr * 100)

        # --- 4. Confidence adjustment ---
        # If low-confidence trades are losing, raise the bar
        low_stats = self.by_conf_band.get("low", self._bucket())
        med_stats = self.by_conf_band.get("med", self._bucket())
        high_stats = self.by_conf_band.get("high", self._bucket())

        low_total = low_stats["wins"] + low_stats["losses"]
        if low_total >= self.min_samples and self._wr(low_stats) < 0.50:
            self.confidence_adj = 0.05  # raise min confidence by 5%
            log.info("   📈 Raising confidence threshold +5%% (low-conf WR: %.1f%%)",
                     self._wr(low_stats) * 100)
        elif low_total >= self.min_samples and self._wr(low_stats) > 0.55:
            self.confidence_adj = -0.03  # lower it slightly — more opportunities
            log.info("   📉 Lowering confidence threshold -3%% (low-conf WR: %.1f%%)",
                     self._wr(low_stats) * 100)
        else:
            self.confidence_adj = 0.0

        # --- 5. Recent momentum — adaptive cooldown ---
        if len(self._recent) >= 10:
            recent_10 = list(self._recent)[-10:]
            recent_wr = sum(1 for t in recent_10 if t["result"] == "win") / 10
            if recent_wr < 0.30:
                self.adaptive_cooldown = 300  # 5 min cooldown — bot is cold
                log.info("   🥶 Recent WR %.0f%% — adding 5min cooldown", recent_wr * 100)
            elif recent_wr < 0.40:
                self.adaptive_cooldown = 120  # 2 min extra cooldown
                log.info("   😐 Recent WR %.0f%% — adding 2min cooldown", recent_wr * 100)
            else:
                self.adaptive_cooldown = 0
                if recent_wr > 0.65:
                    log.info("   🔥 On fire! Recent WR %.0f%% — trading normally", recent_wr * 100)

        log.info("🧠 Review complete. Blocked regimes: %s | Blocked hours: %s | Conf adj: %+.0f%%",
                 self.blocked_regimes or "none",
                 self.blocked_hours or "none",
                 self.confidence_adj * 100)

    # ------------------------------------------------------------------
    def should_trade(self, direction: str, regime: str, confidence: float,
                     hour: int, base_min_conf: float) -> tuple[bool, str]:
        """Check if the adaptive layer allows this trade."""

        if regime in self.blocked_regimes:
            return False, f"Regime '{regime}' blocked by adaptive strategy"

        if hour in self.blocked_hours:
            return False, f"Hour {hour:02d}:00 blocked by adaptive strategy"

        if self.preferred_direction and direction != self.preferred_direction:
            # Don't hard block — just require higher confidence
            adjusted_conf = base_min_conf + 0.08
            if confidence < adjusted_conf:
                return False, (f"Non-preferred direction '{direction}' needs "
                               f"conf ≥{adjusted_conf:.0%}, got {confidence:.0%}")

        effective_min = base_min_conf + self.confidence_adj
        if confidence < effective_min:
            return False, (f"Adaptive conf threshold {effective_min:.0%}, "
                           f"got {confidence:.0%}")

        return True, "OK"

    def get_extra_cooldown(self) -> int:
        """Extra seconds to wait between trades during cold streaks."""
        return self.adaptive_cooldown

    def status_line(self) -> str:
        """Short status for logging."""
        parts = []
        if self.blocked_regimes:
            parts.append(f"⛔reg:{','.join(self.blocked_regimes)}")
        if self.blocked_hours:
            parts.append(f"⛔hr:{','.join(str(h) for h in sorted(self.blocked_hours))}")
        if self.preferred_direction:
            parts.append(f"prefer:{self.preferred_direction}")
        if self.confidence_adj != 0:
            parts.append(f"conf:{self.confidence_adj:+.0%}")
        if self.adaptive_cooldown > 0:
            parts.append(f"cool:{self.adaptive_cooldown}s")
        return " | ".join(parts) if parts else "all-clear"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN BOT
# ═══════════════════════════════════════════════════════════════════════════
class AITradingBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.client: Optional[PocketOptionAsync] = None
        self.candles: deque[Candle] = deque(maxlen=cfg.lookback)
        self.ensemble = EnsemblePredictor()
        self.features_engine = FeatureEngine()
        self.feature_lab = FeatureLab(self.features_engine, review_interval=50)
        self.regime_detector = RegimeDetector()
        self.money_mgr = MoneyManager(cfg)
        self.journal = TradeJournal(cfg.db_path)
        self.perf = PerformanceTracker()
        self.pending_trades: dict[str, TradeRecord] = {}
        self._cooldown_until = 0.0
        self._samples_since_fit = 0
        self._running = False
        self.adaptive = AdaptiveStrategy(review_interval=25, min_samples=15)
        self.expiry_selector = ExpirySelector(expiry_options=cfg.expiry_options)

        # Signal confirmation tracking — candle-based
        self._signal_history: list[tuple[float, Direction, float]] = []  # (candle_ts, dir, conf)
        self._last_trade_time: float = 0.0  # when last trade was placed

    # ------------------------------------------------------------------
    def _load_dataset(self, path: str):
        """Pre-train models on historical CSV data before going live.
        Supports:
          - Standard CSV with headers (time,open,high,low,close,volume)
          - HistData semicolon format: YYYYMMDD HHMMSS;O;H;L;C;V (no headers)
        """
        import csv
        from datetime import datetime as dt

        log.info("Loading dataset from %s …", path)
        candles: list[Candle] = []

        with open(path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline().strip()
            f.seek(0)

            # --- Detect format ---
            if ";" in first_line and not any(
                h in first_line.lower() for h in ["time", "open", "high", "date"]
            ):
                # HistData semicolon-delimited, no headers
                # Format: YYYYMMDD HHMMSS;open;high;low;close;volume
                log.info("Detected HistData semicolon format (no headers)")
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parts = line.split(";")
                        if len(parts) < 5:
                            continue
                        time_str = parts[0].strip()
                        # Parse YYYYMMDD HHMMSS
                        if len(time_str) >= 15:
                            parsed = dt.strptime(time_str, "%Y%m%d %H%M%S")
                        elif len(time_str) >= 8:
                            parsed = dt.strptime(time_str[:8], "%Y%m%d")
                        else:
                            continue
                        ts = parsed.timestamp()
                        candles.append(Candle(
                            timestamp=ts,
                            open=float(parts[1]),
                            high=float(parts[2]),
                            low=float(parts[3]),
                            close=float(parts[4]),
                            volume=float(parts[5]) if len(parts) > 5 else 0.0,
                        ))
                    except (ValueError, TypeError, IndexError):
                        continue
            else:
                # Standard CSV with headers (comma or semicolon)
                delimiter = ";" if ";" in first_line else ","
                reader = csv.DictReader(f, delimiter=delimiter)
                log.info("CSV columns found: %s (delimiter='%s')", reader.fieldnames, delimiter)

                for row in reader:
                    try:
                        time_str = str(row.get("time") or row.get("timestamp")
                                       or row.get("date") or row.get("Date") or "").strip()

                        if "-" in time_str and ":" in time_str:
                            parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                            ts = parsed.timestamp()
                        elif time_str:
                            ts = float(time_str)
                        else:
                            continue

                        volume = (row.get("tick_volume") or row.get("volume")
                                  or row.get("Volume") or row.get("real_volume") or 0)

                        candles.append(Candle(
                            timestamp=ts,
                            open=float(row.get("open") or row.get("Open") or 0),
                            high=float(row.get("high") or row.get("High") or 0),
                            low=float(row.get("low") or row.get("Low") or 0),
                            close=float(row.get("close") or row.get("Close") or 0),
                            volume=float(volume or 0),
                        ))
                    except (ValueError, TypeError, KeyError):
                        continue

        log.info("Parsed %d candles from file.", len(candles))

        if len(candles) < 50:
            log.warning("Dataset too small (%d candles), skipping pre-training.", len(candles))
            return

        # Use ALL candles — more data = better models
        max_candles = 500000
        if len(candles) > max_candles:
            log.info("Dataset has %d candles — using most recent %d.", len(candles), max_candles)
            candles = candles[-max_candles:]

        # Label horizon = default_expiry / timeframe (how many candles = one trade)
        label_horizon = max(1, int(self.cfg.default_expiry / max(self.cfg.timeframe, 1)))
        log.info("Processing %d candles (label horizon = %d candles = %ds) …",
                 len(candles), label_horizon, self.cfg.default_expiry)

        samples_X = []
        samples_y = []
        window = self.cfg.feature_window

        for i in range(max(window, 26), len(candles) - label_horizon):
            # Use candles up to index i for features
            chunk = candles[max(0, i - self.cfg.lookback):i + 1]
            features = self.features_engine.compute(chunk, window)
            if features is None:
                continue

            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Label: did price go up or down over the ACTUAL trade duration?
            future_close = candles[i + label_horizon].close
            current_close = candles[i].close

            if future_close > current_close:
                label = 1  # CALL would have won
            elif future_close < current_close:
                label = 0  # PUT would have won
            else:
                continue  # skip draws

            samples_X.append(features)
            samples_y.append(label)

            # Progress logging
            if len(samples_X) % 5000 == 0:
                log.info("  … generated %d training samples so far", len(samples_X))

        if len(samples_X) < 20:
            log.warning("Only %d valid samples from dataset, skipping.", len(samples_X))
            return

        # --- Walk-forward: Train on 70%, test on 30% ---
        split = int(len(samples_X) * 0.70)
        train_X, train_y = samples_X[:split], samples_y[:split]
        test_X, test_y = samples_X[split:], samples_y[split:]

        log.info("📚 Training on %d samples (70%%) …", len(train_X))

        # Feed training samples to ensemble in batches
        batch = 5000
        for s in range(0, len(train_X), batch):
            e = min(s + batch, len(train_X))
            for x, y in zip(train_X[s:e], train_y[s:e]):
                self.ensemble.add_sample(x, y)
            self.ensemble.partial_fit()

        # Train batch models (GBM + RF) on training data
        self.ensemble.train_batch_models()

        # --- Walk-forward test on 30% ---
        if len(test_X) > 100:
            wins = losses = 0
            for x, true_label in zip(test_X, test_y):
                direction, confidence = self.ensemble.predict(x)
                predicted_call = (direction == Direction.CALL)
                correct = (predicted_call and true_label == 1) or (not predicted_call and true_label == 0)
                if correct:
                    wins += 1
                else:
                    losses += 1
                # Online learning during test
                label = true_label if correct else (1 - true_label)
                self.ensemble.add_sample(x, label)
                if (wins + losses) % 500 == 0:
                    self.ensemble.partial_fit()

            if wins + losses > 0:
                self.ensemble.partial_fit()  # flush remaining

            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            log.info("═" * 50)
            log.info("🧪 WALK-FORWARD TEST (30%%): %d trades", total)
            log.info("   Win: %d | Loss: %d | HONEST WR: %.1f%%", wins, losses, wr)
            breakeven = 1.0 / (1.0 + 0.92) * 100  # 92% payout
            if wr > breakeven:
                log.info("   ✅ EDGE: +%.1f%% above breakeven (%.1f%%)", wr - breakeven, breakeven)
            else:
                log.info("   ❌ Below breakeven (%.1f%%) by %.1f%%", breakeven, breakeven - wr)
            log.info("═" * 50)

        calls = sum(1 for y in samples_y if y == 1)
        puts = sum(1 for y in samples_y if y == 0)
        log.info(
            "✅ Pre-trained on %d samples!  (CALL: %d, PUT: %d)  Models ready.",
            len(samples_X), calls, puts,
        )

        # Save brain after training
        self._save_brain()

    # ------------------------------------------------------------------
    def _reload_from_journal(self):
        """Reload past trades from SQLite and retrain models — survive restarts."""
        past_trades = self.journal.load_completed_trades()
        if not past_trades:
            log.info("No past trades found in journal — starting fresh.")
            return

        loaded = 0
        expected_dim = None
        for t in past_trades:
            try:
                features = np.array(json.loads(t["features_json"]), dtype=np.float64)
                direction = t["direction"]
                result = t["result"]

                # Track expected dimension (use most recent trade's dimension)
                if expected_dim is None:
                    expected_dim = len(features)

                # Skip trades with mismatched feature dimensions
                if len(features) != expected_dim:
                    continue

                dir_int = 1 if direction == "call" else 0
                if result == "win":
                    label = dir_int
                else:
                    label = 1 - dir_int

                self.ensemble.add_sample(features, label)
                loaded += 1

                # Feed feature lab if dimensions match current engine
                expected_total = FeatureEngine.NUM_CORE + FeatureEngine.NUM_EXPERIMENTAL
                if result in ("win", "loss") and len(features) == expected_total:
                    self.feature_lab.record_trade(features, result)
            except Exception:
                continue  # skip corrupted entries

        if loaded > 0:
            self.ensemble.partial_fit()
            log.info(
                "🔄 Reloaded %d trades from journal — models retrained!",
                loaded,
            )

            # Restore performance tracker stats
            wins = sum(1 for t in past_trades if t["result"] == "win")
            losses = sum(1 for t in past_trades if t["result"] == "loss")
            self.perf.wins = wins
            self.perf.losses = losses
            log.info(
                "📊 Restored stats: W:%d L:%d WR:%.1f%%",
                wins, losses, (wins / (wins + losses) * 100) if (wins + losses) > 0 else 50,
            )

            # Feed adaptive strategy from journal
            adaptive_loaded = 0
            for t in past_trades:
                try:
                    result = t.get("result", "")
                    if result not in ("win", "loss"):
                        continue
                    direction = t.get("direction", "call")
                    regime = t.get("regime", "ranging")
                    confidence = float(t.get("confidence", 0.6))
                    entry_time = float(t.get("entry_time", 0))
                    hour = int(datetime.fromtimestamp(
                        entry_time, tz=timezone.utc).hour) if entry_time > 0 else 12
                    self.adaptive.record_trade(direction, regime, confidence, hour, result)
                    adaptive_loaded += 1
                except Exception:
                    continue

            if adaptive_loaded > 0:
                log.info("🧠 Adaptive strategy loaded %d past trades — [%s]",
                         adaptive_loaded, self.adaptive.status_line())

    # ------------------------------------------------------------------
    def _check_indicator_alignment(self, features: np.ndarray, direction: Direction) -> bool:
        """Check if key indicators agree with the ML prediction."""
        rsi = features[14] if len(features) > 14 else 50.0
        macd_hist = features[12] if len(features) > 12 else 0.0
        sma_cross = features[10] if len(features) > 10 else 0.0

        votes = 0
        total = 3

        if direction == Direction.CALL:
            if rsi < 70:       votes += 1  # not overbought
            if macd_hist > 0:  votes += 1  # MACD bullish
            if sma_cross > 0:  votes += 1  # trend up
        else:
            if rsi > 30:       votes += 1  # not oversold
            if macd_hist < 0:  votes += 1  # MACD bearish
            if sma_cross < 0:  votes += 1  # trend down

        aligned = votes >= 2  # at least 2 of 3 must agree
        if not aligned:
            log.debug("Indicator misalignment: %d/%d agree with %s", votes, total, direction.value)
        return aligned

    # ------------------------------------------------------------------
    def _check_signal_ready(self, direction: Direction, confidence: float, candle_ts: float) -> bool:
        """Require N consecutive CANDLES to agree on direction before trading."""

        # Only record once per unique candle
        if self._signal_history and self._signal_history[-1][0] == candle_ts:
            return False  # already checked this candle, wait for next one

        self._signal_history.append((candle_ts, direction, confidence))

        # Keep only recent signals
        max_history = self.cfg.signal_confirmations * 3
        if len(self._signal_history) > max_history:
            self._signal_history = self._signal_history[-max_history:]

        # Check if last N candles all agree on direction
        if len(self._signal_history) < self.cfg.signal_confirmations:
            log.info("📡 Signal building: %d/%d candles agree on %s",
                     len(self._signal_history), self.cfg.signal_confirmations, direction.value)
            return False

        recent = self._signal_history[-self.cfg.signal_confirmations:]
        all_same_dir = all(d == direction for _, d, c in recent)

        if not all_same_dir:
            log.info("📡 Signal not confirmed — mixed directions over last %d candles",
                     self.cfg.signal_confirmations)
            return False

        # Average confidence across confirmations
        avg_conf = np.mean([c for _, _, c in recent])
        log.info("✅ Signal CONFIRMED: %s x%d candles  avg_conf=%.1f%%",
                 direction.value, self.cfg.signal_confirmations, avg_conf * 100)
        return True

    # ------------------------------------------------------------------
    def _save_brain(self):
        """Save all learned state to disk."""
        try:
            self.ensemble.save_brain(self.cfg.brain_path)
            # Save expiry stats alongside brain
            expiry_path = self.cfg.brain_path.replace(".pkl", "_expiry.pkl")
            with open(expiry_path, "wb") as f:
                pickle.dump(self.expiry_selector.save_state(), f)
        except Exception as e:
            log.warning("Failed to save brain: %s", e)

    def _load_brain(self) -> bool:
        """Load pre-trained brain from disk."""
        loaded = self.ensemble.load_brain(self.cfg.brain_path)
        # Also load expiry stats if available
        expiry_path = self.cfg.brain_path.replace(".pkl", "_expiry.pkl")
        if os.path.exists(expiry_path):
            try:
                with open(expiry_path, "rb") as f:
                    self.expiry_selector.load_state(pickle.load(f))
                log.info("⏱ Expiry stats loaded: %s", self.expiry_selector.status_line())
            except Exception as e:
                log.warning("Failed to load expiry stats: %s", e)
        return loaded

    # ------------------------------------------------------------------
    async def start(self):
        """Main entry point."""
        log.info("═" * 60)
        log.info("  🦉 AthenaAI TRADING BOT v5.0 — PocketOption")
        log.info("  Asset: %s  |  Timeframe: %ds", self.cfg.asset, self.cfg.timeframe)
        log.info("  Expiry: AI-selected from %s", [f"{e}s" for e in self.cfg.expiry_options])
        log.info("  Models: SGD + PA + NB + GBM + RF (5-model ensemble)")
        log.info("  Features: %d core + %d experimental  |  Adaptive: ON",
                 FeatureEngine.NUM_CORE, FeatureEngine.NUM_EXPERIMENTAL)
        log.info("═" * 60)

        # Try loading saved brain first
        force_retrain = os.environ.get("PO_RETRAIN", "").strip() == "1"
        brain_loaded = False

        if force_retrain:
            log.info("🔄 Force retrain requested — ignoring saved brain.")
            if os.path.exists(self.cfg.brain_path):
                os.remove(self.cfg.brain_path)
        else:
            brain_loaded = self._load_brain()

        if brain_loaded:
            log.info("✅ Loaded saved brain — skipping dataset training!")
        else:
            # Reload past trades from journal
            try:
                self._reload_from_journal()
            except Exception as e:
                log.error("Failed to reload from journal: %s", e)

            # Pre-train from dataset if provided
            if self.cfg.dataset_path:
                try:
                    self._load_dataset(self.cfg.dataset_path)
                except Exception as e:
                    log.error("Failed to load dataset: %s", e)

        # Connect
        log.info("Connecting to PocketOption …")
        self.client = PocketOptionAsync(ssid=self.cfg.ssid)
        await asyncio.sleep(3)  # allow websocket handshake

        balance = await self.client.balance()
        log.info("Connected!  Balance: $%.2f", balance)

        # Load historical candles
        log.info("Loading %d warmup candles …", self.cfg.warmup_candles)
        raw = await self.client.get_candles(
            self.cfg.asset,
            self.cfg.timeframe,
            self.cfg.warmup_candles,
        )
        for c in raw:
            self.candles.append(self._parse_candle(c))
        log.info("Loaded %d candles.  Starting main loop …", len(self.candles))

        self._running = True
        await asyncio.gather(
            self._candle_stream(),
            self._trade_loop(),
            self._result_checker(),
        )

    # ------------------------------------------------------------------
    async def _candle_stream(self):
        """Subscribe to live candle updates and maintain history."""
        try:
            stream = await self.client.subscribe_symbol(self.cfg.asset)
            async for raw in stream:
                c = self._parse_candle(raw)
                # Only append if new timestamp
                if not self.candles or c.timestamp > self.candles[-1].timestamp:
                    self.candles.append(c)
        except Exception as e:
            log.error("Candle stream error: %s", e)
            self._running = False

    # ------------------------------------------------------------------
    async def _trade_loop(self):
        """Core decision loop with diagnostic logging."""
        await asyncio.sleep(2)  # let candle stream populate
        last_diag = 0  # last diagnostic log time

        while self._running:
            try:
                await asyncio.sleep(self.cfg.poll_interval)
                now = time.time()
                show_diag = (now - last_diag) >= 30  # diagnostic every 30s

                # ── Gate checks with diagnostic ──
                reason = None

                if len(self.candles) < self.cfg.warmup_candles:
                    reason = f"Warming up ({len(self.candles)}/{self.cfg.warmup_candles} candles)"
                elif len(self.pending_trades) >= self.cfg.max_concurrent_trades:
                    reason = f"Max trades open ({len(self.pending_trades)}/{self.cfg.max_concurrent_trades})"
                elif self._last_trade_time > 0 and now - self._last_trade_time < self.cfg.min_wait_between_trades:
                    wait_left = int(self.cfg.min_wait_between_trades - (now - self._last_trade_time))
                    reason = f"Wait between trades ({wait_left}s left)"
                elif not self.money_mgr.can_trade():
                    reason = "Daily loss limit reached"
                elif now < self._cooldown_until:
                    reason = f"Cooldown ({int(self._cooldown_until - now)}s left)"

                if reason:
                    if show_diag: log.info("⏸ %s", reason); last_diag = now
                    continue

                # Extra adaptive cooldown
                extra_cool = self.adaptive.get_extra_cooldown()
                if extra_cool > 0 and self._last_trade_time > 0:
                    if now - self._last_trade_time < extra_cool:
                        if show_diag: log.info("⏸ Adaptive cooldown (%ds)", extra_cool); last_diag = now
                        continue

                # --- feature extraction ---
                candle_list = list(self.candles)
                features = self.features_engine.compute(candle_list, self.cfg.feature_window)
                if features is None:
                    if show_diag: log.info("⏸ Feature compute returned None"); last_diag = now
                    continue

                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                # --- regime ---
                regime = self.regime_detector.detect(candle_list, self.cfg.regime_window)
                if self.cfg.skip_volatile_regime and regime == Regime.VOLATILE:
                    if show_diag: log.info("⏸ Volatile regime — skipping"); last_diag = now
                    continue

                # --- prediction ---
                direction, confidence = self.ensemble.predict(features)

                # --- confidence gate ---
                if confidence < self.cfg.min_confidence:
                    if show_diag:
                        log.info("⏸ Low confidence: %.1f%% (need %.1f%%) dir=%s regime=%s",
                                 confidence * 100, self.cfg.min_confidence * 100,
                                 direction.value, regime.value)
                        last_diag = now
                    self._signal_history.clear()
                    continue

                # --- indicator alignment ---
                if self.cfg.require_indicator_alignment:
                    if not self._check_indicator_alignment(features, direction):
                        if show_diag: log.info("⏸ Indicators misaligned (conf=%.1f%%)", confidence*100); last_diag = now
                        continue

                # --- signal confirmation ---
                current_candle_ts = candle_list[-1].timestamp if candle_list else 0
                if not self._check_signal_ready(direction, confidence, current_candle_ts):
                    if show_diag:
                        log.info("⏸ Signal confirmation (%d/%d) dir=%s conf=%.1f%%",
                                 len(self._signal_history), self.cfg.signal_confirmations,
                                 direction.value, confidence * 100)
                        last_diag = now
                    continue

                self._signal_history.clear()

                # --- consecutive-loss cooldown ---
                if self.perf.consec_losses >= self.cfg.max_consec_losses:
                    self._cooldown_until = now + self.cfg.cooldown_seconds
                    log.warning("Hit %d consecutive losses → cooldown %ds",
                                self.perf.consec_losses, self.cfg.cooldown_seconds)
                    self.perf.consec_losses = 0
                    continue

                # --- adaptive strategy gate ---
                utc_hour = datetime.now(timezone.utc).hour
                can_trade, ad_reason = self.adaptive.should_trade(
                    direction.value, regime.value, confidence,
                    utc_hour, self.cfg.min_confidence,
                )
                if not can_trade:
                    log.info("🧠 Adaptive skip: %s (conf=%.1f%%)", ad_reason, confidence * 100)
                    continue

                # --- stake sizing ---
                stake = self.money_mgr.compute_stake(
                    confidence, self.perf.win_rate, payout=0.85,
                )

                # --- AI EXPIRY SELECTION ---
                chosen_expiry = self.expiry_selector.select(regime, features, confidence)

                # ══════════════════════════════════════
                # ALL GATES PASSED — EXECUTE TRADE!
                # ══════════════════════════════════════
                log.info(
                    "▶ TRADE  %s  $%.2f  conf=%.1f%%  expiry=%ds  regime=%s  [%s]",
                    direction.value.upper(), stake,
                    confidence * 100, chosen_expiry, regime.value,
                    self.expiry_selector.status_line(),
                )

                if direction == Direction.CALL:
                    trade_id, _ = await self.client.buy(
                        self.cfg.asset, stake, chosen_expiry
                    )
                else:
                    trade_id, _ = await self.client.sell(
                        self.cfg.asset, stake, chosen_expiry
                    )

                record = TradeRecord(
                    id=str(trade_id),
                    direction=direction.value,
                    asset=self.cfg.asset,
                    stake=stake,
                    confidence=confidence,
                    regime=regime.value,
                    entry_time=time.time(),
                    expiry=chosen_expiry,
                )
                self.pending_trades[record.id] = record
                self.journal.save_trade(record)
                self._last_trade_time = time.time()

                # Store features for later learning
                record._features = features  # type: ignore[attr-defined]
                record._direction_int = 1 if direction == Direction.CALL else 0  # type: ignore[attr-defined]
                record.features_json = json.dumps(features.tolist())

            except Exception as e:
                log.error("Trade loop error: %s", e, exc_info=True)
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    async def _result_checker(self):
        """Poll pending trades for results and feed them back to the model."""
        while self._running:
            await asyncio.sleep(3)

            resolved = []
            for tid, rec in list(self.pending_trades.items()):
                # Wait at least the trade's chosen expiry before checking
                elapsed = time.time() - rec.entry_time
                if elapsed < rec.expiry + 2:
                    continue

                try:
                    result = await self.client.check_win(tid)
                    # Handle both dict and string responses
                    if isinstance(result, dict):
                        result_str = str(result.get("result", result.get("status", ""))).lower().strip()
                    else:
                        result_str = str(result).lower().strip()

                    log.debug("check_win(%s) raw=%r  parsed=%s", tid, result, result_str)

                    if result_str not in ("win", "loss", "draw"):
                        # Check if result contains the keyword anywhere
                        raw_str = str(result).lower()
                        if "win" in raw_str:
                            result_str = "win"
                        elif "loss" in raw_str or "lose" in raw_str:
                            result_str = "loss"
                        elif "draw" in raw_str:
                            result_str = "draw"
                        else:
                            continue

                    payout = 0.85
                    if result_str == "win":
                        profit = rec.stake * payout
                    elif result_str == "loss":
                        profit = -rec.stake
                    else:
                        profit = 0.0

                    rec.result = result_str
                    rec.profit = profit
                    rec.exit_time = time.time()

                    self.perf.record(result_str, profit)
                    self.money_mgr.record(profit)
                    self.journal.save_trade(rec)

                    # Feed adaptive strategy
                    trade_hour = int(datetime.fromtimestamp(
                        rec.entry_time, tz=timezone.utc).hour)
                    self.adaptive.record_trade(
                        direction=rec.direction,
                        regime=rec.regime,
                        confidence=rec.confidence,
                        hour=trade_hour,
                        result=result_str,
                    )

                    # Feed expiry selector — learn which durations win
                    self.expiry_selector.record_result(rec.expiry, result_str)

                    icon = "✅" if result_str == "win" else ("❌" if result_str == "loss" else "➖")
                    log.info(
                        "%s  %s  $%+.2f  expiry=%ds  |  %s  |  expiry stats: %s",
                        icon, result_str.upper(), profit, rec.expiry,
                        self.perf.summary(), self.expiry_selector.status_line(),
                    )

                    # --- ONLINE LEARNING ---
                    features = getattr(rec, "_features", None)
                    if features is not None and result_str in ("win", "loss"):
                        dir_int = getattr(rec, "_direction_int", 0)
                        if result_str == "win":
                            label = dir_int          # correct prediction
                        else:
                            label = 1 - dir_int      # opposite was correct

                        self.ensemble.add_sample(features, label)
                        self._samples_since_fit += 1

                        # Feed Feature Lab
                        self.feature_lab.record_trade(features, result_str)

                        if self._samples_since_fit >= self.cfg.retrain_every:
                            self.ensemble.partial_fit()
                            self._samples_since_fit = 0

                    # Periodic snapshot
                    if self.perf.total % 10 == 0:
                        regime = self.regime_detector.detect(list(self.candles))
                        self.journal.save_snapshot(
                            self.perf.win_rate, self.perf.total,
                            self.money_mgr.daily_pnl, regime.value,
                        )
                        # Auto-save brain every 10 trades
                        self._save_brain()

                    resolved.append(tid)

                except Exception as e:
                    log.debug("check_win error for %s: %s", tid, e)
                    # If too old, abandon
                    if time.time() - rec.entry_time > rec.expiry * 5:
                        log.warning("Abandoning stale trade %s after timeout: %s", tid, e)
                        resolved.append(tid)

            for tid in resolved:
                self.pending_trades.pop(tid, None)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_candle(raw) -> Candle:
        """Flexible candle parser — handles dict, list, or object."""
        if isinstance(raw, dict):
            return Candle(
                timestamp=float(raw.get("time", raw.get("timestamp", 0)) or 0),
                open=float(raw.get("open", 0) or 0),
                high=float(raw.get("high", 0) or 0),
                low=float(raw.get("low", 0) or 0),
                close=float(raw.get("close", 0) or 0),
                volume=float(raw.get("volume", 0) or 0),
            )
        elif isinstance(raw, (list, tuple)):
            return Candle(
                timestamp=float(raw[0]),
                open=float(raw[1]),
                high=float(raw[2]),
                low=float(raw[3]),
                close=float(raw[4]),
                volume=float(raw[5]) if len(raw) > 5 else 0,
            )
        else:
            return Candle(
                timestamp=float(getattr(raw, "time", getattr(raw, "timestamp", 0)) or 0),
                open=float(getattr(raw, "open", 0) or 0),
                high=float(getattr(raw, "high", 0) or 0),
                low=float(getattr(raw, "low", 0) or 0),
                close=float(getattr(raw, "close", 0) or 0),
                volume=float(getattr(raw, "volume", 0) or 0),
            )

    # ------------------------------------------------------------------
    async def stop(self):
        self._running = False
        self._save_brain()
        log.info("Bot stopped.  Final stats: %s", self.perf.summary())


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # --- Load config from env or defaults ---
    # Expiry options: comma-separated seconds (e.g. "60,120,180,300")
    expiry_str = os.environ.get("PO_EXPIRY_OPTIONS", "60,120,180,300")
    expiry_opts = tuple(int(x.strip()) for x in expiry_str.split(",") if x.strip())

    cfg = BotConfig(
        ssid=os.environ.get("PO_SSID", ""),
        asset=os.environ.get("PO_ASSET", "EURUSD"),
        timeframe=int(os.environ.get("PO_TIMEFRAME", "60")),
        expiry_options=expiry_opts,
        default_expiry=int(os.environ.get("PO_DEFAULT_EXPIRY", "120")),
        base_stake=float(os.environ.get("PO_BASE_STAKE", "10.0")),
        max_stake=float(os.environ.get("PO_MAX_STAKE", "100.0")),
        min_confidence=float(os.environ.get("PO_MIN_CONF", "0.60")),
        max_daily_loss=float(os.environ.get("PO_MAX_DAILY_LOSS", "300.0")),
        dataset_path=os.environ.get("PO_DATASET", ""),
    )

    if not cfg.ssid:
        print("=" * 60)
        print("  ERROR: No SSID provided!")
        print()
        print("  Set your PocketOption session ID:")
        print("    export PO_SSID='your-session-id-here'")
        print()
        print("  Or edit the config in this file.")
        print("=" * 60)
        sys.exit(1)

    bot = AITradingBot(cfg)

    async def run():
        try:
            await bot.start()
        except KeyboardInterrupt:
            await bot.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()