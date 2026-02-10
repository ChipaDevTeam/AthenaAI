from dataclasses import dataclass

@dataclass
class BotConfig:
    """All tuneable knobs in one place."""

    # --- connection ---
    ssid: str = ""                          # PocketOption session ID
    asset: str = "EURUSD"               # trading pair
    timeframe: int = 60                     # candle period (60s for data)

    # --- AI expiry selection ---
    expiry_options: tuple = (120, 300, 600)        # v6: dropped 60s/180s (bad performers)
    default_expiry: int = 300                 # v6: 300s has best WR (69.2%)

    # --- money management ---
    base_stake: float = 25.0                 # minimum trade size ($)
    max_stake: float = 100.0                 # hard ceiling ($)
    kelly_fraction: float = 0.50            # fraction of Kelly to use
    max_daily_loss: float = 300.0            # stop-loss for the day ($)
    max_concurrent_trades: int = 1          # max open trades

    # --- ML & signals ---
    warmup_candles: int = 60                 # candles before first trade
    min_confidence: float = 0.63            # v6: raised from 0.60 — low conf trades lose
    max_confidence: float = 0.85            # v6: NEW — reject overconfident (overfit) signals
    retrain_every: int = 10                 # partial_fit after N new samples
    lookback: int = 200                     # max candle history to keep
    feature_window: int = 20                # rolling window for features

    # --- signal readiness ---
    signal_confirmations: int = 1           # 1 = instant (no multi-candle wait)
    require_indicator_alignment: bool = True # v6: enabled — secondary gate catches bad signals
    skip_volatile_regime: bool = False       # let ML decide
    min_wait_between_trades: int = 60       # 1 min between trades

    # --- trading schedule (UTC hours) ---
    # v6: only trade during statistically profitable hours
    # Based on 546-trade analysis: 08-15 UTC loses money consistently
    trading_hours: tuple = (0, 1, 2, 3, 4, 16, 17, 19, 20, 21, 22, 23)

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
