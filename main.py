import os
import secrets
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Lucidious Nova EA API",
    version="1.0.0",
    description="Backend foundation for MT4 and MT5 trading bridges."
)

API_KEY = os.getenv("LUCIDIOUS_API_KEY", "")
TRADING_ENABLED = False

# Supported markets for Lucidious Nova EA.
# Actual broker symbol names must be verified before trading.

SUPPORTED_MARKETS = {
    "forex": [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
        "EURGBP",
        "EURJPY",
        "GBPJPY",
        "USDZAR",
    ],
    "metals": [
        "XAUUSD",  # Gold
        "XAGUSD",  # Silver
    ],
    "crypto": [
        "BTCUSD",  # Bitcoin
    ],
    "indices": [
        "NAS100",  # Nasdaq 100; broker naming may vary
    ],
}


def check_api_key(provided_key: str | None):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured on the server."
        )

    if not provided_key or not secrets.compare_digest(
        provided_key, API_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid API key.")


class TradingCommand(BaseModel):
    platform: str = Field(pattern="^(MT4|MT5)$")
    action: str = Field(pattern="^(start|stop)$")


@app.get("/")
def home():
    return {
        "name": "Lucidious Nova EA API",
        "status": "online",
        "trading": "disabled_until_bridge_is_connected",
        "supported_platforms": ["MT4", "MT5"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/status")
def status(x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    return {
        "api": "online",
        "mt4_bridge": "not_connected",
        "mt5_bridge": "not_connected",
        "live_trading_enabled": False,
        "message": "Connect and verify a trading bridge before enabling orders."
    }


@app.post("/api/trading")
def trading_command(
    command: TradingCommand,
    x_api_key: str | None = Header(default=None)
):
    check_api_key(x_api_key)

    if command.action == "start":
        raise HTTPException(
            status_code=409,
            detail=(
                "Trading cannot start yet. A verified "
                f"{command.platform} bridge must be connected first."
            )
        )

    return {
        "accepted": True,
        "platform": command.platform,
        "action": "stop",
        "message": "Trading remains disabled until a bridge is implemented."
}
@app.get("/api/signal")
def get_signal(
    symbol: str,
    platform: str,
    x_api_key: str | None = Header(default=None)
):
    check_api_key(x_api_key)

    platform = platform.upper()

    if platform not in ("MT4", "MT5"):
        raise HTTPException(
            status_code=400,
            detail="Platform must be MT4 or MT5."
        )

    if not symbol.strip():
        raise HTTPException(
            status_code=400,
            detail="A trading symbol is required."
        )

    return {
        "platform": platform,
        "symbol": symbol.upper(),
        "signal": "NONE",
        "message": "Waiting for a verified trading signal."
    }
  
from typing import List
from pydantic import BaseModel, Field

class CandleData(BaseModel):
    closes: List[float] = Field(min_length=22, max_length=500)

class StrategyRequest(BaseModel):
    symbol: str
    platform: str = Field(pattern="^(MT4|MT5)$")
    closes: List[float] = Field(min_length=22, max_length=500)

def calculate_ema(values: List[float], period: int) -> List[float]:
    multiplier = 2 / (period + 1)
    ema = values[0]
    result = [ema]

    for price in values[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
        result.append(ema)

    return result

@app.post("/api/strategy/signal")
def strategy_signal(
    request: StrategyRequest,
    x_api_key: str | None = Header(default=None)
):
    check_api_key(x_api_key)

    closes = request.closes

    if any(price <= 0 for price in closes):
        raise HTTPException(
            status_code=400,
            detail="Closing prices must be greater than zero."
        )

    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)

    previous = "BUY" if ema9[-2] > ema21[-2] else "SELL" if ema9[-2] < ema21[-2] else "NONE"
    current = "BUY" if ema9[-1] > ema21[-1] else "SELL" if ema9[-1] < ema21[-1] else "NONE"

    signal = "NONE"
    if previous == "SELL" and current == "BUY":
        signal = "BUY"
    elif previous == "BUY" and current == "SELL":
        signal = "SELL"

    return {
        "symbol": request.symbol.upper(),
        "platform": request.platform,
        "strategy": "EMA_9_21_CROSSOVER",
        "signal": signal,
        "ema9": round(ema9[-1], 6),
        "ema21": round(ema21[-1], 6),
        "message": "Signal calculated from supplied closing prices; verify before trading."
    }
  
from typing import List
from pydantic import BaseModel, Field

class OHLCData(BaseModel):
    time: List[str] = Field(min_length=10, max_length=500)
    highs: List[float] = Field(min_length=10, max_length=500)
    lows: List[float] = Field(min_length=10, max_length=500)
    closes: List[float] = Field(min_length=10, max_length=500)
    symbol: str
    platform: str = Field(pattern="^(MT4|MT5)$")

@app.post("/api/strategy/smc-ict")
def smc_ict_signal(
    request: OHLCData,
    x_api_key: str | None = Header(default=None)
):
    check_api_key(x_api_key)

    highs = request.highs
    lows = request.lows
    closes = request.closes
    n = len(closes)

    if not (len(highs) == len(lows) == len(closes) == len(request.time)):
        raise HTTPException(
            status_code=400,
            detail="Time, high, low and close arrays must have equal lengths."
        )

    if any(
        low <= 0 or high < low or close <= 0
        or close > high or close < low
        for high, low, close in zip(highs, lows, closes)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid OHLC candle values."
        )

    # Basic recent swing levels (simple local extrema)
    swing_highs = [
        i for i in range(1, n - 1)
        if highs[i] > highs[i - 1] and highs[i] >= highs[i + 1]
    ]
    swing_lows = [
        i for i in range(1, n - 1)
        if lows[i] < lows[i - 1] and lows[i] <= lows[i + 1]
    ]

    last_high = highs[swing_highs[-1]] if swing_highs else None
    last_low = lows[swing_lows[-1]] if swing_lows else None

    previous_close = closes[-2]
    latest_close = closes[-1]

    bullish_bos = last_high is not None and previous_close <= last_high < latest_close
    bearish_bos = last_low is not None and previous_close >= last_low > latest_close

    # Basic liquidity sweep: wick breaches a prior swing and closes back inside
    bullish_sweep = (
        last_low is not None
        and lows[-1] < last_low
        and latest_close > last_low
    )
    bearish_sweep = (
        last_high is not None
        and highs[-1] > last_high
        and latest_close < last_high
    )

    # Three-candle fair value gap approximation
    bullish_fvg = lows[-1] > highs[-3]
    bearish_fvg = highs[-1] < lows[-3]

    # Simplified CHoCH proxy: sweep against the latest directional break
    choch = "NONE"
    if bullish_sweep and bearish_bos:
        choch = "BULLISH"
    elif bearish_sweep and bullish_bos:
        choch = "BEARISH"

    signal = "NONE"
    reasons = []

    if bullish_bos:
        reasons.append("BULLISH_BOS")
    if bearish_bos:
        reasons.append("BEARISH_BOS")
    if bullish_sweep:
        reasons.append("BULLISH_LIQUIDITY_SWEEP")
    if bearish_sweep:
        reasons.append("BEARISH_LIQUIDITY_SWEEP")
    if bullish_fvg:
        reasons.append("BULLISH_FVG")
    if bearish_fvg:
        reasons.append("BEARISH_FVG")
    if choch != "NONE":
        reasons.append("CHOCH_" + choch)

    # Conservative initial rule: require a sweep plus a matching structure break
    if bullish_sweep and bullish_bos:
        signal = "BUY"
    elif bearish_sweep and bearish_bos:
        signal = "SELL"

    return {
        "symbol": request.symbol.upper(),
        "platform": request.platform,
        "strategy": "SMC_ICT_BASIC",
        "signal": signal,
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "choch": choch,
        "bullish_liquidity_sweep": bullish_sweep,
        "bearish_liquidity_sweep": bearish_sweep,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "reasons": reasons,
        "message": "Preliminary OHLC analysis only; verify before trading."
    }

@app.get("/api/markets")
def get_supported_markets(
    x_api_key: str | None = Header(default=None)
):
    check_api_key(x_api_key)

    return {
        "name": "Lucidious Nova EA",
        "markets": SUPPORTED_MARKETS,
        "trading_enabled": TRADING_ENABLED,
        "message": "Market list configured. Live prices and order execution are not connected yet."
    }
