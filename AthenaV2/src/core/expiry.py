import numpy as np
from ..utils.enums import Regime

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
