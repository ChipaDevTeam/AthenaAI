from collections import deque

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
