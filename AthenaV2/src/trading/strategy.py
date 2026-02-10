from collections import deque
from typing import Optional
from ..utils.logger import log

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
