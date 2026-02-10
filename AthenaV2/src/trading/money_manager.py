from datetime import datetime, timezone
from ..config import BotConfig
from ..utils.logger import log

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
        # v6: Gentler confidence scaling (old formula crushed stakes to ~$1-3)
        # 60% conf → 0.50×, 65% → 0.65×, 70% → 0.80×, 75% → 0.95×
        stake *= max(0.5, (confidence - 0.45) * 2.0)
        stake = max(self.cfg.base_stake, min(stake, self.cfg.max_stake))
        return round(stake, 2)

    def record(self, pnl: float):
        self.daily_pnl += pnl
