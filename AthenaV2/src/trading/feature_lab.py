import numpy as np
from ..utils.logger import log
from ..core.feature_engine import FeatureEngine

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
