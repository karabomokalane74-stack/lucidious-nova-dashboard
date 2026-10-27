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
  
