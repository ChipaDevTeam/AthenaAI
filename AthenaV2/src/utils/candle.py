from dataclasses import dataclass
from typing import Optional
from .enums import Regime

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

@dataclass
class TradeRecord:
    id: str
    direction: str
    asset: str
    stake: float
    confidence: float
    regime: str
    entry_time: float
    expiry: int = 120                      # AI-chosen expiry in seconds
    result: Optional[str] = None   # "win" / "loss" / "draw"
    profit: Optional[float] = None
    exit_time: Optional[float] = None
    features_json: Optional[str] = None  # stored feature vector as JSON
