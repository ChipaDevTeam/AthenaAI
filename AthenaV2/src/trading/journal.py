import sqlite3
import time
import json
import numpy as np
from ..utils.candle import TradeRecord

class TradeJournal:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          TEXT PRIMARY KEY,
                direction   TEXT,
                asset       TEXT,
                stake       REAL,
                confidence  REAL,
                regime      TEXT,
                entry_time  REAL,
                exit_time   REAL,
                result      TEXT,
                profit      REAL,
                features    TEXT,
                expiry      INTEGER
            )
        """)
        # Migrate old DB: add columns if missing
        for col, ctype in [("features", "TEXT"), ("expiry", "INTEGER")]:
            try:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {ctype}")
            except Exception:
                pass  # column already exists
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_snapshots (
                ts          REAL,
                win_rate    REAL,
                total_trades INTEGER,
                daily_pnl   REAL,
                regime      TEXT
            )
        """)
        self.conn.commit()

    def save_trade(self, t: TradeRecord):
        self.conn.execute(
            "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.id, t.direction, t.asset, t.stake, t.confidence,
             t.regime, t.entry_time, t.exit_time, t.result, t.profit,
             t.features_json, t.expiry),
        )
        self.conn.commit()

    def load_completed_trades(self) -> list[dict]:
        """Load all completed trades with features for retraining."""
        cur = self.conn.execute(
            "SELECT direction, result, features FROM trades "
            "WHERE result IN ('win', 'loss') AND features IS NOT NULL "
            "ORDER BY entry_time ASC"
        )
        rows = cur.fetchall()
        trades = []
        for direction, result, features_json in rows:
            trades.append({
                "direction": direction,
                "result": result,
                "features_json": features_json,
            })
        return trades

    def save_snapshot(self, win_rate, total, daily_pnl, regime):
        self.conn.execute(
            "INSERT INTO model_snapshots VALUES (?,?,?,?,?)",
            (time.time(), win_rate, total, daily_pnl, regime),
        )
        self.conn.commit()

    def recent_win_rate(self, n: int = 50) -> float:
        cur = self.conn.execute(
            "SELECT result FROM trades WHERE result IS NOT NULL ORDER BY entry_time DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
        if not rows:
            return 0.5
        wins = sum(1 for r in rows if r[0] == "win")
        return wins / len(rows)

    def total_trades(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM trades")
        return cur.fetchone()[0]
