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
