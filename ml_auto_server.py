import asyncio
import csv
import json
import logging
import math
import random
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import websockets

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("QuantexMLServer")

BROKER_ENDPOINTS = {
    "pocket_option": "wss://api-eu.po.market/socket.io/?EIO=4&transport=websocket",
    "quotex": "wss://://qxbroker.com",
}

PAIRS = [
    "EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "AUDCAD_otc",
    "EURGBP_otc", "USDCHF_otc", "EURJPY_otc", "GBPJPY_otc",
    "NZDJPY_otc", "USDBRL_otc", "USDINR_otc", "BTCUSD_otc",
]

BASE_PRICES = {
    "EURUSD_otc": 1.09320, "GBPUSD_otc": 1.28052, "USDJPY_otc": 154.243,
    "AUDCAD_otc": 0.91271, "EURGBP_otc": 0.87214, "USDCHF_otc": 0.88857,
    "EURJPY_otc": 169.577, "GBPJPY_otc": 195.452, "NZDJPY_otc": 92.239,
    "USDBRL_otc": 5.46420, "USDINR_otc": 82.0150, "BTCUSD_otc": 67338.9,
}

FEATURE_COLS = [
    "ker_10", "atr_ratio", "zlema_diff", "macd_slope",
    "rsi_7", "rsi_2", "bb_pct_b", "vwap_z",
    "wick_rejection", "tick_delta", "tps", "body_overlap",
]

HISTORY_DIR = Path("history")
LOGS_DIR = Path("logs")


@dataclass
class Candle:
    minute_ts: int
    open: float
    high: float
    low: float
    close: float
    up_ticks: int = 1
    down_ticks: int = 1
    total_ticks: int = 2
    last_price: float = 0.0

    def update(self, price: float):
        if price > self.last_price:
            self.up_ticks += 1
        elif price < self.last_price:
            self.down_ticks += 1
        self.total_ticks += 1
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.last_price = price

    @property
    def tick_delta(self) -> float:
        return (self.up_ticks - self.down_ticks) / max(1, self.total_ticks)


@dataclass
class ModelBundle:
    models: Dict[str, object] = field(default_factory=dict)
    accuracies: Dict[str, float] = field(default_factory=lambda: {"catboost": 0.56, "xgboost": 0.55, "lightgbm": 0.54})
    brier_scores: Dict[str, float] = field(default_factory=lambda: {"catboost": 0.24, "xgboost": 0.245, "lightgbm": 0.25})
    top_dna: List[str] = field(default_factory=lambda: ["rsi_2", "vwap_z"])
    last_trained: float = 0.0
    is_trained: bool = False


class AuditLogger:
    """Persistent SQLite and CSV logging for candles, predictions, and resolved trades."""

    def __init__(self, db_path: Path = LOGS_DIR / "quantex_audit.db", csv_path: Path = LOGS_DIR / "trades_audit.csv"):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.csv_path = csv_path
        self._init_db()
        self._init_csv()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    minute_ts INTEGER NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    up_ticks INTEGER,
                    down_ticks INTEGER,
                    source TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resolved_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    regime TEXT,
                    top_dna TEXT,
                    stake REAL,
                    payout_pct REAL,
                    pnl REAL,
                    status TEXT NOT NULL
                )
            """)
            conn.commit()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc", "pair", "direction", "entry_price", "close_price",
                    "confidence", "regime", "top_dna", "stake", "payout_pct", "pnl", "status"
                ])

    def log_candle(self, pair: str, candle: Candle, source: str = "live"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO candles (pair, minute_ts, open, high, low, close, up_ticks, down_ticks, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair, candle.minute_ts, candle.open, candle.high, candle.low,
                 candle.close, candle.up_ticks, candle.down_ticks, source)
            )
            conn.commit()

    def log_resolved_trade(self, trade: dict):
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        dna_str = "+".join(trade.get("top_dna", []))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO resolved_trades "
                "(timestamp_utc, pair, direction, entry_price, close_price, confidence, regime, top_dna, stake, payout_pct, pnl, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts_str, trade["pair"], trade["direction"], trade["entry_price"],
                    trade["close_price"], trade["confidence"], trade["regime"],
                    dna_str, trade["stake"], trade["payout_pct"], trade["pnl"], trade["status"]
                )
            )
            conn.commit()

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts_str, trade["pair"], trade["direction"], trade["entry_price"],
                trade["close_price"], trade["confidence"], trade["regime"],
                dna_str, trade["stake"], trade["payout_pct"], trade["pnl"], trade["status"]
            ])

    def fetch_recent_trades(self, limit: int = 100) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM resolved_trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]


class MicroIntelligenceEngine:
    def __init__(self, audit_logger: AuditLogger):
        self.audit = audit_logger
        self.candles: Dict[str, deque] = {p: deque(maxlen=1500) for p in PAIRS}
        self.active_candle: Dict[str, Candle] = {}
        self.champions: Dict[str, ModelBundle] = {p: ModelBundle() for p in PAIRS}
        self.pending_signals: Dict[str, dict] = {}
        self.confidence_gate: float = 60.0
        self.default_stake: float = 10.0
        self.default_payout: float = 0.92
        self._load_or_seed_history()

    def _load_or_seed_history(self):
        """Load historical 1m CSV files from ./history/<PAIR>.csv; pad with synthetic history if < 240 rows."""
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        now_min = int(time.time() // 60) * 60

        for pair in PAIRS:
            csv_file = HISTORY_DIR / f"{pair}.csv"
            loaded_count = 0

            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = [c.strip().lower() for c in df.columns]
                    col_map = {"time": "timestamp", "ts": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close"}
                    df = df.rename(columns=col_map)

                    if {"open", "high", "low", "close"}.issubset(df.columns):
                        df = df.tail(1200).reset_index(drop=True)
                        for idx, row in df.iterrows():
                            ts_val = (
                                int(row["timestamp"])
                                if "timestamp" in df.columns and not pd.isna(row["timestamp"])
                                else now_min - ((len(df) - idx) * 60)
                            )
                            if ts_val > 1e12:
                                ts_val //= 1000
                            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                            up = int(row["up_ticks"]) if "up_ticks" in df.columns else random.randint(15, 45)
                            dn = int(row["down_ticks"]) if "down_ticks" in df.columns else random.randint(15, 45)
                            self.candles[pair].append(Candle(ts_val, o, h, l, c, up, dn, up + dn, c))
                            loaded_count += 1
                        logger.info(f"[{pair}] Backfilled {loaded_count} real 1m candles from {csv_file}")
                except Exception as exc:
                    logger.warning(f"[{pair}] Failed reading {csv_file}: {exc}. Falling back to bootstrap.")

            if loaded_count < 240:
                needed = 240 - loaded_count
                price = self.candles[pair][0].open if loaded_count > 0 else BASE_PRICES[pair]
                vol = price * 0.00035
                synthetic_prefix = []

                for i in range(needed, 0, -1):
                    ts = now_min - ((loaded_count + i) * 60)
                    o = price
                    c = o + np.random.normal(0, vol)
                    h = max(o, c) + abs(np.random.normal(0, vol * 0.6))
                    l = min(o, c) - abs(np.random.normal(0, vol * 0.6))
                    up, dn = random.randint(15, 45), random.randint(15, 45)
                    synthetic_prefix.append(Candle(ts, o, h, l, c, up, dn, up + dn, c))
                    price = c
                existing = list(self.candles[pair])
                self.candles[pair].clear()
                self.candles[pair].extend(synthetic_prefix + existing)
            last_close = self.candles[pair][-1].close
            self.active_candle[pair] = Candle(now_min, last_close, last_close, last_close, last_close, 1, 1, 2, last_close)

    def ingest_tick(self, pair: str, price: float, ts_ms: int, source: str = "live") -> List[dict]:
        """Update active 1m candle, resolve expired trades on minute boundary, and generate live prediction."""
        if pair not in self.candles:
            return []
        events: List[dict] = []
        sec = ts_ms // 1000
        minute_ts = (sec // 60) * 60
        sec_into_candle = sec % 60
        curr = self.active_candle.get(pair)
        if curr is None:
            curr = Candle(minute_ts, price, price, price, price, 1, 1, 2, price)
            self.active_candle[pair] = curr
        elif minute_ts > curr.minute_ts:
            # 1. Finalize completed candle and persist to SQLite
            self.candles[pair].append(curr)
            self.audit.log_candle(pair, curr, source=source)
            # 2. Resolve pending signal that expired at the end of this candle
            if pair in self.pending_signals:
                sig = self.pending_signals.pop(pair)
                won = (curr.close > sig["entry_price"]) if sig["direction"] == "CALL" else (curr.close < sig["entry_price"])
                pnl = round(self.default_stake * self.default_payout, 2) if won else -round(self.default_stake, 2)
                resolved_record = {
                    "type": "trade_resolved",
                    "pair": pair,
                    "direction": sig["direction"],
                    "entry_price": sig["entry_price"],
                    "close_price": curr.close,
                    "confidence": sig["confidence"],
                    "regime": sig["regime"],
                    "top_dna": sig["top_dna"],
                    "stake": self.default_stake,
                    "payout_pct": round(self.default_payout * 100, 1),
                    "pnl": pnl,
                    "status": "WIN" if won else "LOSS",
                }
                self.audit.log_resolved_trade(resolved_record)
                events.append(resolved_record)
                logger.info(f"[{pair}] Trade Resolved -> {resolved_record['status']} ({pnl:+.2f}$)")
            # 3. Evaluate auto-signal trigger for the new candle
            latest_pred = self.predict_live(pair, curr, 60)
            if (
                latest_pred["direction"] in ("CALL", "PUT")
                and latest_pred["raw_confidence"] >= self.confidence_gate
                and latest_pred["regime"] != "CHOPPY"
            ):
                self.pending_signals[pair] = {
                    "direction": latest_pred["direction"],
                    "entry_price": price,
                    "confidence": latest_pred["raw_confidence"],
                    "regime": latest_pred["regime"],
                    "top_dna": latest_pred["top_dna"],
                }
                events.append({
                    "type": "signal_fired",
                    "pair": pair,
                    "direction": latest_pred["direction"],
                    "entry_price": price,
                    "confidence": latest_pred["raw_confidence"],
                })
            # 4. Initialize new 1m candle
            curr = Candle(minute_ts, price, price, price, price, 1, 1, 2, price)
            self.active_candle[pair] = curr
        else:
            curr.update(price)
        events.append(self.predict_live(pair, curr, sec_into_candle))
        return events

    def _compute_features(self, history: List[Candle]) -> pd.DataFrame:
        df = pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low, "close": c.close,
            "tick_delta": c.tick_delta, "total_ticks": c.total_ticks,
        } for c in history])
        close, high, low, op = df["close"], df["high"], df["low"], df["open"]
        # 1. Kaufman Efficiency Ratio (KER 10)
        change_10 = (close - close.shift(10)).abs()
        volatility_10 = close.diff().abs().rolling(10).sum().replace(0, 1e-9)
        df["ker_10"] = (change_10 / volatility_10).fillna(0.3)
        # 2. ATR Squeeze Ratio (ATR 7 / ATR 21)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_7 = tr.rolling(7).mean()
        atr_21 = tr.rolling(21).mean().replace(0, 1e-9)
        df["atr_ratio"] = (atr_7 / atr_21).fillna(1.0)
        # 3. Zero-Lag EMA Cross (ZLEMA 5 / 13)
        zlema_5 = (2 * close - close.shift(2)).ewm(span=5).mean()
        zlema_13 = (2 * close - close.shift(6)).ewm(span=13).mean()
        df["zlema_diff"] = ((zlema_5 - zlema_13) / atr_7.replace(0, 1e-9)).fillna(0.0)
        # 4. Impulse MACD Slope (6/13/5)
        macd = close.ewm(span=6).mean() - close.ewm(span=13).mean()
        hist = macd - macd.ewm(span=5).mean()
        df["macd_slope"] = (hist.diff() / atr_7.replace(0, 1e-9)).fillna(0.0)
        # 5 & 6. Micro-RSI 7 & RSI 2
        def calc_rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().replace(0, 1e-9)
            return 100 - (100 / (1 + (gain / loss)))
        df["rsi_7"] = calc_rsi(close, 7).fillna(50.0)
        df["rsi_2"] = calc_rsi(close, 2).fillna(50.0)
        # 7. Bollinger %B (20 / 2.2)
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std().replace(0, 1e-9)
        upper_bb = sma_20 + 2.2 * std_20
        lower_bb = sma_20 - 2.2 * std_20
        df["bb_pct_b"] = ((close - lower_bb) / (upper_bb - lower_bb).replace(0, 1e-9)).fillna(0.5)
        # 8. Rolling 60m Tick-Weighted VWAP Z-Score
        typical = (high + low + close) / 3.0
        vwap_60 = (typical * df["total_ticks"]).rolling(60).sum() / df["total_ticks"].rolling(60).sum().replace(0, 1)
        vwap_std = (close - vwap_60).rolling(60).std().replace(0, 1e-9)
        df["vwap_z"] = ((close - vwap_60) / vwap_std).fillna(0.0)
        # 9. Wick Rejection Ratio
        rng = (high - low).replace(0, 1e-9)
        upper_wick = high - pd.concat([op, close], axis=1).max(axis=1)
        lower_wick = pd.concat([op, close], axis=1).min(axis=1) - low
        df["wick_rejection"] = ((lower_wick - upper_wick) / rng).fillna(0.0)
        # 10 & 11. Tick Delta & TPS
        df["tps"] = (df["total_ticks"] / 60.0).fillna(1.0)
        # 12. 10-Candle Body Overlap (Fractal Chop)
        body_hi = pd.concat([op, close], axis=1).max(axis=1)
        body_lo = pd.concat([op, close], axis=1).min(axis=1)
        overlap = (
            pd.concat([body_hi, body_hi.shift(1)], axis=1).min(axis=1)
            - pd.concat([body_lo, body_lo.shift(1)], axis=1).max(axis=1)
        ).clip(lower=0)
        df["body_overlap"] = (overlap / rng).rolling(10).mean().fillna(0.3)
        return df[FEATURE_COLS].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def train_pair(self, pair: str):
        hist = list(self.candles[pair])
        if len(hist) < 120:
            return
        X_all = self._compute_features(hist)
        closes = np.array([c.close for c in hist])
        opens = np.array([c.open for c in hist])
        next_diff = closes[1:] - opens[1:]
        valid_mask = np.abs(next_diff) > (closes[:-1] * 1e-6)
        X = X_all.iloc[:-1][valid_mask].reset_index(drop=True)
        y = (next_diff[valid_mask] > 0).astype(int)
        if len(X) < 80 or len(np.unique(y)) < 2:
            return
        split_idx = int(len(X) * 0.75)
        X_train, y_train = X.iloc[:split_idx], y[:split_idx]
        X_val, y_val = X.iloc[split_idx + 3:], y[split_idx + 3:]
        if len(X_val) < 15 or len(np.unique(y_val)) < 2:
            return
        n_train = len(X_train)
        sample_weights = np.array([0.988 ** (n_train - 1 - i) for i in range(n_train)])
        candidates = {
            "catboost": CatBoostClassifier(iterations=120, depth=4, learning_rate=0.06, verbose=0, random_seed=42),
            "xgboost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.06, eval_metric="logloss", random_state=42),
            "lightgbm": LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.06, verbose=-1, random_state=42),
        }
        new_bundle = ModelBundle(last_trained=time.time(), is_trained=True)
        for name, clf in candidates.items():
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            probs = clf.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            new_bundle.models[name] = clf
            new_bundle.accuracies[name] = round(float(accuracy_score(y_val, preds)), 4)
            new_bundle.brier_scores[name] = round(float(brier_score_loss(y_val, probs)), 4)
        importances = new_bundle.models["catboost"].get_feature_importance()
        top_idx = np.argsort(importances)[::-1][:2]
        new_bundle.top_dna = [FEATURE_COLS[i] for i in top_idx]
        old_bundle = self.champions[pair]
        if not old_bundle.is_trained or np.mean(list(new_bundle.accuracies.values())) >= np.mean(list(old_bundle.accuracies.values())) - 0.01:
            self.champions[pair] = new_bundle
            logger.info(
                f"[{pair}] Trained -> CatBoost: {new_bundle.accuracies['catboost']*100:.1f}% | "
                f"XGB: {new_bundle.accuracies['xgboost']*100:.1f}% | "
                f"LGBM: {new_bundle.accuracies['lightgbm']*100:.1f}% | DNA: {new_bundle.top_dna}"
            )

    def predict_live(self, pair: str, active: Candle, sec_into_candle: int) -> dict:
        seq = list(self.candles[pair])[-120:] + [active]
        feats_df = self._compute_features(seq)
        latest = feats_df.iloc[-1:]
        row = latest.iloc[0]
        ker = float(row["ker_10"])
        atr_ratio = float(row["atr_ratio"])
        if ker < 0.22 or atr_ratio < 0.65:
            regime = "CHOPPY"
        elif ker >= 0.45:
            regime = "TRENDING"
        else:
            regime = "MEAN-REVERTING"
        bundle = self.champions[pair]
        model_probs_up = {}
        weights = {}
        if bundle.is_trained:
            for name, clf in bundle.models.items():
                p_up = float(clf.predict_proba(latest)[0][1])
                model_probs_up[name] = round(p_up * 100, 1)
                weights[name] = max(0.01, (bundle.accuracies[name] - 0.48) ** 2)
        else:
            model_probs_up = {"catboost": 50.0, "xgboost": 50.0, "lightgbm": 50.0}
            weights = {"catboost": 1.0, "xgboost": 1.0, "lightgbm": 1.0}
        total_w = sum(weights.values())
        blended_up = sum(model_probs_up[m] * weights[m] for m in weights) / total_w
        if regime == "CHOPPY":
            blended_up = 50.0 + (blended_up - 50.0) * 0.55
        if float(row["vwap_z"]) > 2.0 and blended_up > 50.0:
            blended_up = 48.0
        elif float(row["vwap_z"]) < -2.0 and blended_up < 50.0:
            blended_up = 52.0
        direction = "CALL" if blended_up >= 50.0 else "PUT"
        raw_confidence = blended_up if direction == "CALL" else (100.0 - blended_up)
        return {
            "type": "prediction",
            "pair": pair,
            "price": active.close,
            "seconds_remaining": max(0, 60 - sec_into_candle),
            "direction": "WAIT" if regime == "CHOPPY" else direction,
            "raw_confidence": round(raw_confidence, 2),
            "regime": regime,
            "top_dna": bundle.top_dna,
            "models": {
                m: {
                    "prob_up": model_probs_up[m],
                    "accuracy": round(bundle.accuracies[m] * 100, 1),
                    "brier": bundle.brier_scores[m],
                }
                for m in ["catboost", "xgboost", "lightgbm"]
            },
            "indicators": {
                "ker_10": round(ker, 3),
                "zlema_diff": round(float(row["zlema_diff"]), 2),
                "macd_slope": round(float(row["macd_slope"]), 2),
                "rsi_7": round(float(row["rsi_7"]), 1),
                "rsi_2": round(float(row["rsi_2"]), 1),
                "bb_pct_b": round(float(row["bb_pct_b"]), 2),
                "vwap_z": round(float(row["vwap_z"]), 2),
                "tick_delta": round(float(row["tick_delta"]), 2),
            },
        }


class QuantexServer:
    def __init__(self):
        self.audit = AuditLogger()
        self.engine = MicroIntelligenceEngine(self.audit)
        self.clients: set[Any] = set()
        self.broker_ws: Optional[Any] = None
        self._broker_lock = asyncio.Lock()
        self.live_mode = False
        self.broker = "pocket_option"
        self.ssid = ""

    async def broadcast(self, payload: dict):
        if not self.clients:
            return
        msg = json.dumps(payload)
        await asyncio.gather(*(c.send(msg) for c in list(self.clients)), return_exceptions=True)

    async def auto_train_loop(self):
        loop = asyncio.get_running_loop()
        logger.info("Running initial walk-forward ML training across all 12 OTC pairs...")
        for pair in PAIRS:
            await loop.run_in_executor(None, self.engine.train_pair, pair)
        while True:
            await asyncio.sleep(600)
            logger.info("Running scheduled 10-minute Auto-Train cycle...")
            for pair in PAIRS:
                await loop.run_in_executor(None, self.engine.train_pair, pair)

    async def simulator_loop(self):
        while True:
            if not self.live_mode:
                now_ms = int(time.time() * 1000)
                for pair in PAIRS:
                    curr = self.engine.active_candle[pair]
                    step = np.random.normal(0, curr.close * 0.00008)
                    new_price = round(curr.close + step, 5 if curr.close < 10 else 3)
                    events = self.engine.ingest_tick(pair, new_price, now_ms, source="simulated")
                    for ev in events:
                        await self.broadcast(ev)
            await asyncio.sleep(1.0)

    async def connect_broker(self, broker: str, ssid: str):
        async with self._broker_lock:
            await self._disconnect_broker()
            self.broker = broker
            self.ssid = ssid.strip()
            url = BROKER_ENDPOINTS.get(broker)
            if not url:
                await self.broadcast({"type": "error", "message": f"Unknown broker: {broker}"})
                return
            try:
                ws = await websockets.connect(
                    url,
                    origin="https://pocketoption.com" if broker == "pocket_option" else "https://qxbroker.com",
                    ping_interval=None,
                )
                self.broker_ws = ws
                self.live_mode = True
                await self.broadcast({"type": "status", "status": "connecting", "broker": broker})
                asyncio.create_task(self._listen_broker(ws, broker, self.ssid))
            except Exception as exc:
                self.live_mode = False
                await self.broadcast({"type": "error", "message": str(exc)})

    async def disconnect_broker(self, announce: bool = False):
        async with self._broker_lock:
            await self._disconnect_broker()
            if announce:
                await self.broadcast({"type": "status", "status": "disconnected"})

    async def _disconnect_broker(self):
        self.live_mode = False
        ws, self.broker_ws = self.broker_ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _listen_broker(self, ws: Any, broker: str, ssid: str):
        connected = False
        try:
            async for raw in ws:
                if self.broker_ws is not ws:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")
                if raw.startswith("0"):
                    await ws.send("40")
                    await asyncio.sleep(0.2)
                    auth_frame = (
                        ssid if ssid.startswith('42["auth"')
                        else f'42["auth",{json.dumps({"session": ssid, "isDemo": 1, "uid": 0, "platform": 2})}]'
                    )
                    await ws.send(auth_frame)
                    await asyncio.sleep(0.3)
                    for p in PAIRS:
                        await ws.send(f'42["changeSymbol",{{"asset":"{p}","period":60}}]')
                        await asyncio.sleep(0.05)
                elif raw == "2":
                    await ws.send("3")
                elif raw.startswith("42"):
                    try:
                        data = json.loads(raw[2:])
                        tick = None
                        if isinstance(data, list) and len(data) > 0:
                            if isinstance(data[0], list) and len(data[0]) >= 3:
                                tick = data[0]
                            elif len(data) >= 3 and isinstance(data[0], str):
                                tick = data
                        if tick is not None:
                            pair, ts, price = str(tick[0]), float(tick[1]), float(tick[2])
                            if pair not in self.engine.candles or not math.isfinite(ts) or not math.isfinite(price) or ts <= 0 or price <= 0:
                                continue
                            ts_ms = int(ts * 1000) if ts < 1e12 else int(ts)
                            async with self._broker_lock:
                                if self.broker_ws is not ws:
                                    break
                                events = self.engine.ingest_tick(pair, price, ts_ms, source="live_ssid")
                                if not connected:
                                    connected = True
                                    await self.broadcast({"type": "status", "status": "connected", "broker": broker})
                                for ev in events:
                                    await self.broadcast(ev)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning(f"Live broker feed closed: {exc}")
        finally:
            async with self._broker_lock:
                if self.broker_ws is ws:
                    self.broker_ws = None
                    self.live_mode = False
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    await self.broadcast({"type": "status", "status": "disconnected"})

    async def handler(self, ws: Any):
        self.clients.add(ws)
        logger.info(f"Terminal UI connected ({len(self.clients)} active)")
        try:
            async for message in ws:
                cmd = json.loads(message)
                action = cmd.get("action")
                if action == "connect":
                    await self.connect_broker(cmd.get("broker", "pocket_option"), cmd.get("ssid", ""))
                elif action == "disconnect":
                    await self.disconnect_broker(announce=True)
                elif action == "set_gate":
                    self.engine.confidence_gate = float(cmd.get("gate", 60.0))
                elif action == "set_stake":
                    self.engine.default_stake = float(cmd.get("stake", 10.0))
                elif action == "get_history":
                    rows = self.audit.fetch_recent_trades(int(cmd.get("limit", 100)))
                    await ws.send(json.dumps({"type": "history_dump", "trades": rows}))
                elif action == "retrain":
                    loop = asyncio.get_running_loop()
                    for pair in PAIRS:
                        await loop.run_in_executor(None, self.engine.train_pair, pair)
                    await self.broadcast({"type": "status", "status": "retrained"})
                elif action == "ping":
                    await ws.send(json.dumps({"type": "pong", "ts": int(time.time() * 1000)}))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)


async def main():
    server = QuantexServer()
    asyncio.create_task(server.auto_train_loop())
    asyncio.create_task(server.simulator_loop())
    logger.info("QUANTEX ML Auto-Train & Audit Server running on ws://127.0.0.1:8765")
    logger.info("SQLite DB: ./logs/quantex_audit.db | CSV Trade Log: ./logs/trades_audit.csv")
    async with websockets.serve(server.handler, "127.0.0.1", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
