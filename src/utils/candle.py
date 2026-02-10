from dataclasses import dataclass

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

def parse_candle(raw) -> Candle:
    """Flexible candle parser — handles dict, list, or object."""
    if isinstance(raw, dict):
        return Candle(
            timestamp=float(raw.get("time", raw.get("timestamp", 0)) or 0),
            open=float(raw.get("open", 0) or 0),
            high=float(raw.get("high", 0) or 0),
            low=float(raw.get("low", 0) or 0),
            close=float(raw.get("close", 0) or 0),
            volume=float(raw.get("volume", 0) or 0),
        )
    elif isinstance(raw, (list, tuple)):
        return Candle(
            timestamp=float(raw[0]),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]) if len(raw) > 5 else 0,
        )
    else:
        return Candle(
            timestamp=float(getattr(raw, "time", getattr(raw, "timestamp", 0)) or 0),
            open=float(getattr(raw, "open", 0) or 0),
            high=float(getattr(raw, "high", 0) or 0),
            low=float(getattr(raw, "low", 0) or 0),
            close=float(getattr(raw, "close", 0) or 0),
            volume=float(getattr(raw, "volume", 0) or 0),
        )
