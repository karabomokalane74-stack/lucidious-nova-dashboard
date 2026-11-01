import math
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

APP_VERSION = "2.1.0"
API_KEY = os.getenv("LUCIDIOUS_API_KEY", "").strip()
TRADING_ENABLED = False

DASHBOARD_ORIGIN = "https://karabomokalane74-stack.github.io"

app = FastAPI(
    title="Lucidious Nova EA API",
    version=APP_VERSION,
    description="Market analysis API. No live trade execution.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[DASHBOARD_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
    max_age=600,
)

SUPPORTED_PLATFORMS = ["MT4", "MT5"]

STANDARD_MARKETS = {
    "forex": [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
        "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "USDZAR",
    ],
    "metals": ["XAUUSD", "XAGUSD"],
    "crypto": ["BTCUSD", "ETHUSD"],
    "indices": ["NAS100", "US30", "GER40", "UK100", "SPX500", "AUS200"],
    "energies": ["USOIL", "UKOIL", "NATGAS"],
}

BROKER_MARKETS = {"MT4": {}, "MT5": {}}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check_api_key(provided_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server authentication is not configured.",
        )

    if not provided_key or not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TradingCommand(StrictModel):
    platform: str = Field(pattern=r"^(MT4|MT5)$")
    action: str = Field(pattern=r"^(start|stop)$")


class StrategyRequest(StrictModel):
    symbol: str = Field(min_length=1, max_length=100)
    platform: str = Field(pattern=r"^(MT4|MT5)$")
    mode: str = Field(default="SWING", pattern=r"^(SCALP|DAY|SWING)$")
    highs: list[float] = Field(min_length=200, max_length=2000)
    lows: list[float] = Field(min_length=200, max_length=2000)
    closes: list[float] = Field(min_length=200, max_length=2000)

    @field_validator("symbol")
    @classmethod
    def clean_symbol(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Symbol cannot be blank.")
        return value


class BrokerInstrument(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    base_name: Optional[str] = Field(default=None, max_length=100)
    description: Optional[str] = Field(default=None, max_length=250)
    category: Optional[str] = Field(default=None, max_length=30)
    digits: Optional[int] = Field(default=None, ge=0, le=12)
    volume_min: Optional[float] = Field(default=None, gt=0, le=1000000)
    volume_max: Optional[float] = Field(default=None, gt=0, le=1000000)
    volume_step: Optional[float] = Field(default=None, gt=0, le=1000000)
    trade_allowed: bool = False

    @field_validator("name", "base_name", "category")
    @classmethod
    def clean_optional_names(cls, value):
        return value.strip() if value is not None else value


class BrokerMarketUpdate(StrictModel):
    broker_name: str = Field(min_length=1, max_length=100)
    platform: str = Field(pattern=r"^(MT4|MT5)$")
    account_currency: Optional[str] = Field(
        default=None, min_length=3, max_length=10
    )
    instruments: list[BrokerInstrument] = Field(
        min_length=1, max_length=5000
    )

    @field_validator("broker_name")
    @classmethod
    def clean_broker_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Broker name cannot be blank.")
        return value


def validate_candles(highs, lows, closes):
    if not (len(highs) == len(lows) == len(closes)):
        raise HTTPException(
            status_code=400,
            detail="High, low and close arrays must have equal lengths.",
        )

    for high, low, close in zip(highs, lows, closes):
        if not all(math.isfinite(x) for x in (high, low, close)):
            raise HTTPException(
                status_code=400,
                detail="Candle values must be finite numbers.",
            )

        if (
            low <= 0 or high <= 0 or close <= 0
            or high < low or not low <= close <= high
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid OHLC candle values.",
            )


def calculate_ema(values, period):
    if not values or period < 1:
        raise ValueError("EMA requires values and a positive period.")

    multiplier = 2 / (period + 1)
    ema = values[0]
    result = [ema]

    for price in values[1:]:
        ema = price * multiplier + ema * (1 - multiplier)
        result.append(ema)

    return result


def calculate_rsi(values, period=14):
    if len(values) <= period:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            avg_gain * (period - 1) + gains[i]
        ) / period
        avg_loss = (
            avg_loss * (period - 1) + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(highs, lows, closes, period=14):
    true_ranges = []

    for i in range(1, len(closes)):
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    if len(true_ranges) < period:
        return 0.0

    atr = sum(true_ranges[:period]) / period

    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period

    return atr


def detect_market_regime(closes):
    ema50 = calculate_ema(closes, 50)
    ema200 = calculate_ema(closes, 200)

    current_price = closes[-1]
    fast = ema50[-1]
    slow = ema200[-1]

    if current_price > fast > slow:
        return "BULLISH"

    if current_price < fast < slow:
        return "BEARISH"

    return "RANGING"


def ema_trend_strategy(closes):
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    ema50 = calculate_ema(closes, 50)

    if ema9[-1] > ema21[-1] > ema50[-1]:
        return "BUY", "EMA bullish alignment"

    if ema9[-1] < ema21[-1] < ema50[-1]:
        return "SELL", "EMA bearish alignment"

    return "HOLD", "No clear EMA alignment"


def trend_pullback_strategy(closes):
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)

    price = closes[-1]
    previous = closes[-2]

    if ema20[-1] > ema50[-1] and previous <= ema20[-2] and price > ema20[-1]:
        return "BUY", "Bullish EMA pullback"

    if ema20[-1] < ema50[-1] and previous >= ema20[-2] and price < ema20[-1]:
        return "SELL", "Bearish EMA pullback"

    return "HOLD", "No pullback confirmation"


def donchian_strategy(highs, lows, closes, period=20):
    upper = max(highs[-period - 1:-1])
    lower = min(lows[-period - 1:-1])
    price = closes[-1]

    if price > upper:
        return "BUY", "Donchian upside breakout"

    if price < lower:
        return "SELL", "Donchian downside breakout"

    return "HOLD", "No Donchian breakout"


def rsi_mean_reversion_strategy(closes):
    rsi = calculate_rsi(closes)

    if rsi < 30:
        return "BUY", "RSI oversold condition"

    if rsi > 70:
        return "SELL", "RSI overbought condition"

    return "HOLD", "RSI is neutral"


def atr_breakout_strategy(highs, lows, closes):
    atr = calculate_atr(highs, lows, closes)

    if atr <= 0:
        return "HOLD", "ATR unavailable"

    recent_high = max(highs[-20:-1])
    recent_low = min(lows[-20:-1])
    price = closes[-1]

    if price > recent_high and price - recent_high >= atr * 0.1:
        return "BUY", "ATR-confirmed upside breakout"

    if price < recent_low and recent_low - price >= atr * 0.1:
        return "SELL", "ATR-confirmed downside breakout"

    return "HOLD", "No ATR breakout"


def smc_ict_strategy(highs, lows, closes):
    recent_high = max(highs[-20:-1])
    recent_low = min(lows[-20:-1])
    previous_high = max(highs[-40:-20])
    previous_low = min(lows[-40:-20])

    price = closes[-1]

    if recent_low < previous_low and price > previous_low:
        return "BUY", "Possible sell-side liquidity sweep"

    if recent_high > previous_high and price < previous_high:
        return "SELL", "Possible buy-side liquidity sweep"

    return "HOLD", "No clear liquidity sweep"


def run_strategy_bundle(highs, lows, closes):
    strategies = {
        "EMA_TREND": ema_trend_strategy(closes),
        "TREND_PULLBACK": trend_pullback_strategy(closes),
        "DONCHIAN": donchian_strategy(highs, lows, closes),
        "RSI_MEAN_REVERSION": rsi_mean_reversion_strategy(closes),
        "ATR_BREAKOUT": atr_breakout_strategy(highs, lows, closes),
        "SMC_ICT": smc_ict_strategy(highs, lows, closes),
    }

    buy_count = sum(1 for signal, _ in strategies.values() if signal == "BUY")
    sell_count = sum(1 for signal, _ in strategies.values() if signal == "SELL")

    if buy_count > sell_count and buy_count >= 2:
        overall = "BUY"
    elif sell_count > buy_count and sell_count >= 2:
        overall = "SELL"
    else:
        overall = "HOLD"

    return {
        "overall_signal": overall,
        "buy_votes": buy_count,
        "sell_votes": sell_count,
        "strategies": {
            name: {"signal": signal, "reason": reason}
            for name, (signal, reason) in strategies.items()
        },
    }


def select_strategy_by_regime(regime):
    if regime in ("BULLISH", "BEARISH"):
        return ["EMA_TREND", "TREND_PULLBACK", "DONCHIAN", "SMC_ICT"]

    return ["RSI_MEAN_REVERSION", "ATR_BREAKOUT", "SMC_ICT"]


def build_analysis(request):
    validate_candles(request.highs, request.lows, request.closes)

    regime = detect_market_regime(request.closes)
    bundle = run_strategy_bundle(
        request.highs,
        request.lows,
        request.closes,
    )

    return {
        "symbol": request.symbol,
        "platform": request.platform,
        "mode": request.mode,
        "market_regime": regime,
        "recommended_strategies": select_strategy_by_regime(regime),
        "price": request.closes[-1],
        "rsi": round(calculate_rsi(request.closes), 2),
        "atr": round(
            calculate_atr(
                request.highs,
                request.lows,
                request.closes,
            ),
            8,
        ),
        "analysis": bundle,
        "live_execution": False,
        "disclaimer": "Analysis only. No live trade was placed.",
        "timestamp": utc_now(),
    }@app.get("/")
def root():
    return {
        "name": "Lucidious Nova EA API",
        "version": APP_VERSION,
        "status": "online",
        "trading_enabled": TRADING_ENABLED,
        "live_execution": False,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Lucidious Nova EA API",
        "timestamp": utc_now(),
    }


@app.get("/api/public-config")
def public_config():
    return {
        "app_name": "Lucidious Nova EA",
        "api_version": APP_VERSION,
        "trading_enabled": TRADING_ENABLED,
        "supported_platforms": SUPPORTED_PLATFORMS,
        "live_execution": False,
        "authentication_configured": bool(API_KEY),
    }


@app.get("/api/status")
def api_status():
    return {
        "api": "online",
        "version": APP_VERSION,
        "trading_enabled": TRADING_ENABLED,
        "mt4_connected": False,
        "mt5_connected": False,
        "live_execution": False,
        "message": "No trading terminal bridge is connected.",
        "timestamp": utc_now(),
    }


@app.get("/api/markets")
def get_standard_markets():
    return {
        "markets": STANDARD_MARKETS,
        "source": "standard catalogue",
        "live_broker_data": False,
    }


@app.get("/api/broker/markets")
def get_broker_markets(
    platform: str = Query(default="MT5", pattern=r"^(MT4|MT5)$"),
    x_api_key: Optional[str] = Header(default=None),
):
    check_api_key(x_api_key)

    return {
        "platform": platform,
        "markets": BROKER_MARKETS[platform],
        "source": "temporary in-memory catalogue",
        "terminal_connected": False,
        "live_broker_data": False,
    }


@app.post("/api/broker/markets")
def update_broker_markets(
    update: BrokerMarketUpdate,
    x_api_key: Optional[str] = Header(default=None),
):
    check_api_key(x_api_key)

    names = [item.name.upper() for item in update.instruments]

    if len(names) != len(set(names)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate instrument names are not allowed.",
        )

    for instrument in update.instruments:
        if (
            instrument.volume_min is not None
            and instrument.volume_max is not None
            and instrument.volume_min > instrument.volume_max
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid volume limits for {instrument.name}.",
            )

    BROKER_MARKETS[update.platform] = {
        "broker_name": update.broker_name,
        "account_currency": update.account_currency,
        "instruments": [
            item.model_dump() for item in update.instruments
        ],
        "updated_at": utc_now(),
    }

    return {
        "success": True,
        "message": "Temporary instrument catalogue updated.",
        "platform": update.platform,
        "instrument_count": len(update.instruments),
        "terminal_connected": False,
        "live_broker_data": False,
    }


@app.post("/api/analyze")
def analyze_market(
    request: StrategyRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    check_api_key(x_api_key)
    return build_analysis(request)


@app.post("/api/strategy")
def strategy_analysis(
    request: StrategyRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    check_api_key(x_api_key)
    return build_analysis(request)


@app.get("/api/strategies")
def get_strategies():
    return {
        "strategies": [
            "EMA_TREND",
            "SMC_ICT",
            "TREND_PULLBACK",
            "DONCHIAN",
            "RSI_MEAN_REVERSION",
            "ATR_BREAKOUT",
            "AUTO_REGIME_SELECTION",
        ],
        "description": "Rule-based analysis strategies. Not profit-guaranteed.",
    }


@app.get("/api/platforms")
def get_platforms():
    return {
        "platforms": SUPPORTED_PLATFORMS,
        "connected": False,
        "message": "MT4 and MT5 bridges are not implemented.",
    }@app.post("/api/trading/start")
def start_trading(
    command: TradingCommand,
    x_api_key: Optional[str] = Header(default=None),
):
    check_api_key(x_api_key)

    raise HTTPException(
        status_code=409,
        detail=(
            "Trading is disabled. No MT4 or MT5 bridge is connected. "
            "No trade has been placed."
        ),
    )


@app.post("/api/trading/stop")
def stop_trading(
    command: TradingCommand,
    x_api_key: Optional[str] = Header(default=None),
):
    check_api_key(x_api_key)

    return {
        "success": True,
        "trading": False,
        "message": "No live trading engine is running.",
    }


@app.get("/api/trading/status")
def trading_status():
    return {
        "trading_enabled": TRADING_ENABLED,
        "running": False,
        "mt4_connected": False,
        "mt5_connected": False,
        "live_execution": False,
        "message": "Trading engine is not connected.",
    }


@app.get("/api/account")
def account_status(
    x_api_key: Optional[str] = Header(default=None),
):
    check_api_key(x_api_key)

    return {
        "connected": False,
        "balance": None,
        "equity": None,
        "currency": None,
        "message": "No broker account is connected.",
    }


@app.get("/api/safety")
def safety_status():
    return {
        "trading_enabled": TRADING_ENABLED,
        "live_execution": False,
        "mt4_bridge_connected": False,
        "mt5_bridge_connected": False,
        "safe_to_execute": False,
        "message": "Live execution is unavailable.",
    }


@app.get("/api/time")
def server_time():
    return {
        "utc": utc_now(),
        "timezone": "UTC",
    }
