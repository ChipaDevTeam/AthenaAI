from enum import Enum

class Direction(Enum):
    CALL = "call"
    PUT = "put"

class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
