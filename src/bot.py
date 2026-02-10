import asyncio
import json
import os
import time
import sys
import pickle
import numpy as np
from collections import deque
from datetime import datetime, timezone
from typing import Optional

# 3rd party
try:
    from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
except ImportError:
    print("ERROR: BinaryOptionsToolsV2 not installed.")
    print("Install with:  pip install binaryoptionstoolsv2")
    sys.exit(1)

# Local imports
from src.config import BotConfig
from src.constants import Direction, Regime
from src.utils.logger import log
from src.utils.candle import Candle, parse_candle
from src.core.feature_engine import FeatureEngine
from src.core.models import EnsemblePredictor
from src.core.regime import RegimeDetector
from src.core.expiry import ExpirySelector
from src.trading.money_manager import MoneyManager
from src.trading.journal import TradeJournal
from src.trading.performance import PerformanceTracker
from src.trading.feature_lab import FeatureLab
from src.trading.strategy import AdaptiveStrategy
from src.trading.trade import TradeRecord


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
            self.candles.append(parse_candle(c))
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
                c = parse_candle(raw)
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
    async def stop(self):
        self._running = False
        self._save_brain()
        log.info("Bot stopped.  Final stats: %s", self.perf.summary())
