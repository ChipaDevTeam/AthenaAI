import numpy as np
import math
from datetime import datetime, timezone
from typing import Optional
from ..utils.candle import Candle

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
