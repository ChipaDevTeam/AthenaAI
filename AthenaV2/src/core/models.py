import os
import pickle
import numpy as np

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

from ..utils.logger import log
from ..utils.enums import Direction

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

        # Batch models vote (v6: REDUCED weight — they overfit and produce anti-correlated confidence)
        if self._batch_fitted:
            for m in self._batch_models:
                w = m["accuracy_ema"] * 0.75  # v6: was 2.0 — demoted due to overfitting
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

        # v6: Cap confidence — data shows >78% confidence = overfit noise (50% WR)
        p = min(p, 0.78)
        p = max(p, 0.22)

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
