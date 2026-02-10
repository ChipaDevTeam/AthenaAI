import numpy as np
from src.constants import Regime
from src.utils.candle import Candle

class RegimeDetector:
    @staticmethod
    def detect(candles: list[Candle], window: int = 30) -> Regime:
        if len(candles) < window:
            return Regime.RANGING

        closes = np.array([c.close for c in candles[-window:]])
        rets = np.diff(closes) / (closes[:-1] + 1e-10)

        trend = np.polyfit(np.arange(len(closes)), closes, 1)[0]
        vol = np.std(rets)
        # mean_ret = np.mean(rets)

        # Normalise trend by price level
        rel_trend = trend / (closes[-1] + 1e-10) * window

        if vol > 0.005:
            return Regime.VOLATILE
        if rel_trend > 0.002:
            return Regime.TRENDING_UP
        if rel_trend < -0.002:
            return Regime.TRENDING_DOWN
        return Regime.RANGING
