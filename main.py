import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

app = FastAPI(
    title="Lucidious Nova EA API",
    version="2.0.0",
    description="Multi-strategy trading analysis and broker discovery API."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("LUCIDIOUS_API_KEY", "")

# Live trading remains disabled until a verified MT5 bridge,
# risk controls and execution safeguards are implemented.
TRADING_ENABLED = False

SUPPORTED_PLATFORMS = ["MT4", "MT5"]

STANDARD_MARKETS = {
    "forex": [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "USDCHF", "NZDUSD", "EURGBP",
        "EURJPY", "GBPJPY", "USDZAR"
    ],
    "metals": [
        "XAUUSD", "XAGUSD"
    ],
    "crypto": [
        "BTCUSD", "ETHUSD"
    ],
    "indices": [
        "NAS100", "US30", "GER40", "UK100",
        "SPX500", "AUS200"
    ],
    "energies": [
        "USOIL", "UKOIL", "NATGAS"
    ]
}

# Broker instruments received from the MT4/MT5 bridge.
# These dictionaries are temporary in-memory storage.
BROKER_MARKETS = {
    "MT4": {},
    "MT5": {}
}


# =========================================================
# 2. AUTHENTICATION
# =========================================================

def check_api_key(provided_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured on the server."
        )

    if not provided_key or not secrets.compare_digest(
        provided_key, API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )


# =========================================================
# 3. DATA MODELS
# =========================================================

class TradingCommand(BaseModel):
    platform: str = Field(pattern="^(MT4|MT5)$")
    action: str = Field(pattern="^(start|stop)$")


class StrategyRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=100)

    platform: str = Field(pattern="^(MT4|MT5)$")

    mode: str = Field(
        default="SWING",
        pattern="^(SCALP|DAY|SWING)$"
    )

    highs: list[float] = Field(
        min_length=50,
        max_length=2000
    )

    lows: list[float] = Field(
        min_length=50,
        max_length=2000
    )

    closes: list[float] = Field(
        min_length=50,
        max_length=2000
    )


class BrokerInstrument(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    base_name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None

    digits: Optional[int] = None

    volume_min: Optional[float] = None
    volume_max: Optional[float] = None
    volume_step: Optional[float] = None

    trade_allowed: bool = False


class BrokerMarketUpdate(BaseModel):
    broker_name: str = Field(min_length=1, max_length=100)

    platform: str = Field(pattern="^(MT4|MT5)$")

    account_currency: Optional[str] = None

    instruments: list[BrokerInstrument] = Field(
        min_length=1,
        max_length=10000
    )


# =========================================================
# 4. INPUT VALIDATION
# =========================================================

def validate_candles(highs, lows, closes):

    if not (
        len(highs) == len(lows) == len(closes)
    ):
        raise HTTPException(
            status_code=400,
            detail="High, low and close arrays must have equal lengths."
        )

    for high, low, close in zip(highs, lows, closes):

        if (
            low <= 0
            or high <= 0
            or close <= 0
            or high < low
            or close > high
            or close < low
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid OHLC candle values."
            )


# =========================================================
# 5. TECHNICAL INDICATORS
# =========================================================

def calculate_ema(values, period):

    multiplier = 2 / (period + 1)

    ema = values[0]
    result = [ema]

    for price in values[1:]:

        ema = (
            price * multiplier
            + ema * (1 - multiplier)
        )

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
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_atr(highs, lows, closes, period=14):

    true_ranges = []

    for i in range(1, len(closes)):

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

        true_ranges.append(tr)

    if len(true_ranges) < period:
        return 0.0

    atr = sum(true_ranges[:period]) / period

    for tr in true_ranges[period:]:

        atr = (
            atr * (period - 1) + tr
        ) / period

    return atr


def calculate_adx(highs, lows, closes, period=14):

    if len(closes) < period * 2 + 1:
        return 0.0

    plus_dm = []
    minus_dm = []
    true_ranges = []

    for i in range(1, len(closes)):

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        plus_dm.append(
            up_move
            if up_move > down_move and up_move > 0
            else 0
        )

        minus_dm.append(
            down_move
            if down_move > up_move and down_move > 0
            else 0
        )

        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        ))

    atr = sum(true_ranges[:period])
    plus = sum(plus_dm[:period])
    minus = sum(minus_dm[:period])

    dx_values = []

    for i in range(period, len(true_ranges)):

        atr = atr - atr / period + true_ranges[i]

        plus = plus - plus / period + plus_dm[i]
        minus = minus - minus / period + minus_dm[i]

        if atr <= 0:
            continue

        plus_di = 100 * plus / atr
        minus_di = 100 * minus / atr

        denominator = plus_di + minus_di

        if denominator == 0:
            dx_values.append(0)
        else:
            dx_values.append(
                100 * abs(plus_di - minus_di) / denominator
            )

    if not dx_values:
        return 0.0

    return sum(dx_values) / len(dx_values)


def calculate_donchian(highs, lows, period=20):

    if len(highs) <= period:
        return None, None

    # Exclude the latest candle to avoid using its own
    # high or low as the breakout boundary.
    upper = max(highs[-period-1:-1])
    lower = min(lows[-period-1:-1])

    return upper, lower


# =========================================================
# 6. STRATEGY 1: EMA TREND FOLLOWING
# =========================================================

def ema_trend_strategy(closes):

    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    ema50 = calculate_ema(closes, 50)

    signal = "NONE"

    if (
        ema9[-1] > ema21[-1] > ema50[-1]
        and closes[-1] > ema50[-1]
    ):
        signal = "BUY"

    elif (
        ema9[-1] < ema21[-1] < ema50[-1]
        and closes[-1] < ema50[-1]
    ):
        signal = "SELL"

    return {
        "strategy": "EMA_TREND_FOLLOWING",
        "signal": signal,
        "ema9": ema9[-1],
        "ema21": ema21[-1],
        "ema50": ema50[-1]
    }


# =========================================================
# 7. STRATEGY 2: SMC / ICT
# =========================================================

def smc_ict_strategy(highs, lows, closes):

    n = len(closes)

    swing_highs = [
        i for i in range(1, n - 1)
        if highs[i] > highs[i - 1]
        and highs[i] >= highs[i + 1]
    ]

    swing_lows = [
        i for i in range(1, n - 1)
        if lows[i] < lows[i - 1]
        and lows[i] <= lows[i + 1]
    ]

    last_high = (
        highs[swing_highs[-1]]
        if swing_highs else None
    )

    last_low = (
        lows[swing_lows[-1]]
        if swing_lows else None
    )

    previous_close = closes[-2]
    latest_close = closes[-1]

    bullish_bos = (
        last_high is not None
        and previous_close <= last_high < latest_close
    )

    bearish_bos = (
        last_low is not None
        and previous_close >= last_low > latest_close
    )

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

    bullish_fvg = lows[-1] > highs[-3]
    bearish_fvg = highs[-1] < lows[-3]

    signal = "NONE"

    if bullish_sweep and bullish_bos:
        signal = "BUY"

    elif bearish_sweep and bearish_bos:
        signal = "SELL"

    return {
        "strategy": "SMC_ICT_BASIC",
        "signal": signal,
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "bullish_sweep": bullish_sweep,
        "bearish_sweep": bearish_sweep,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg
    }


# =========================================================
# 8. STRATEGY 3: TREND PULLBACK
# =========================================================

def trend_pullback_strategy(closes):

    ema21 = calculate_ema(closes, 21)
    ema50 = calculate_ema(closes, 50)
    ema200 = calculate_ema(closes, 200)

    signal = "NONE"

    bullish_trend = ema50[-1] > ema200[-1]
    bearish_trend = ema50[-1] < ema200[-1]

    bullish_pullback = (
        bullish_trend
        and closes[-2] < ema21[-2]
        and closes[-1] > ema21[-1]
    )

    bearish_pullback = (
        bearish_trend
        and closes[-2] > ema21[-2]
        and closes[-1] < ema21[-1]
    )

    if bullish_pullback:
        signal = "BUY"

    elif bearish_pullback:
        signal = "SELL"

    return {
        "strategy": "TREND_PULLBACK",
        "signal": signal
    }


# =========================================================
# 9. STRATEGY 4: DONCHIAN BREAKOUT
# =========================================================

def donchian_breakout_strategy(highs, lows, closes):

    upper, lower = calculate_donchian(
        highs, lows, 20
    )

    signal = "NONE"

    if upper is None:
        return {
            "strategy": "DONCHIAN_BREAKOUT",
            "signal": signal
        }

    if closes[-1] > upper:
        signal = "BUY"

    elif closes[-1] < lower:
        signal = "SELL"

    return {
        "strategy": "DONCHIAN_BREAKOUT",
        "signal": signal,
        "upper_channel": upper,
        "lower_channel": lower
    }


# =========================================================
# 10. STRATEGY 5: RSI MEAN REVERSION
# =========================================================

def mean_reversion_strategy(closes):

    rsi = calculate_rsi(closes, 14)

    signal = "NONE"

    # Require RSI to cross back out of an extreme zone.
    previous_rsi = calculate_rsi(closes[:-1], 14)

    if previous_rsi < 30 and rsi >= 30:
        signal = "BUY"

    elif previous_rsi > 70 and rsi <= 70:
        signal = "SELL"

    return {
        "strategy": "RSI_MEAN_REVERSION",
        "signal": signal,
        "rsi": rsi
    }


# =========================================================
# 11. STRATEGY 6: ATR VOLATILITY BREAKOUT
# =========================================================

def atr_breakout_strategy(highs, lows, closes):

    atr = calculate_atr(highs, lows, closes, 14)

    signal = "NONE"

    if atr <= 0:
        return {
            "strategy": "ATR_VOLATILITY_BREAKOUT",
            "signal": signal,
            "atr": atr
        }

    recent_range = highs[-1] - lows[-1]

    previous_atr = calculate_atr(
        highs[:-1],
        lows[:-1],
        closes[:-1],
        14
    )

    # Require volatility expansion and a directional close.
    if (
        previous_atr > 0
        and recent_range > previous_atr * 1.5
    ):

        if closes[-1] > highs[-2]:
            signal = "BUY"

        elif closes[-1] < lows[-2]:
            signal = "SELL"

    return {
        "strategy": "ATR_VOLATILITY_BREAKOUT",
        "signal": signal,
        "atr": atr,
        "candle_range": recent_range
    }


# =========================================================
# 12. MARKET REGIME DETECTION
# =========================================================

def detect_market_regime(highs, lows, closes):

    adx = calculate_adx(highs, lows, closes)

    ema50 = calculate_ema(closes, 50)
    ema200 = calculate_ema(closes, 200)

    atr = calculate_atr(highs, lows, closes)

    if atr <= 0:
        regime = "UNKNOWN"

    elif adx >= 25:

        if ema50[-1] > ema200[-1]:
            regime = "BULLISH_TREND"

        elif ema50[-1] < ema200[-1]:
            regime = "BEARISH_TREND"

        else:
            regime = "TRENDING"

    elif adx < 20:
        regime = "RANGING"

    else:
        regime = "TRANSITIONAL"

    return {
        "regime": regime,
        "adx": adx,
        "atr": atr
    }


# =========================================================
# 13. INTELLIGENT STRATEGY SELECTOR
# =========================================================

def select_strategy(
    regime,
    ema,
    smc,
    pullback,
    donchian,
    mean_reversion,
    atr_breakout
):

    if regime == "BULLISH_TREND":

        if smc["signal"] == "BUY":
            return "SMC_ICT_BASIC", "BUY"

        if pullback["signal"] == "BUY":
            return "TREND_PULLBACK", "BUY"

        if ema["signal"] == "BUY":
            return "EMA_TREND_FOLLOWING", "BUY"

        if donchian["signal"] == "BUY":
            return "DONCHIAN_BREAKOUT", "BUY"

    elif regime == "BEARISH_TREND":

        if smc["signal"] == "SELL":
            return "SMC_ICT_BASIC", "SELL"

        if pullback["signal"] == "SELL":
            return "TREND_PULLBACK", "SELL"

        if ema["signal"] == "SELL":
            return "EMA_TREND_FOLLOWING", "SELL"

        if donchian["signal"] == "SELL":
            return "DONCHIAN_BREAKOUT", "SELL"

    elif regime == "RANGING":

        if mean_reversion["signal"] != "NONE":
            return (
                "RSI_MEAN_REVERSION",
                mean_reversion["signal"]
            )

        if smc["signal"] != "NONE":
            return "SMC_ICT_BASIC", smc["signal"]

    elif regime == "TRANSITIONAL":

        if atr_breakout["signal"] != "NONE":
            return (
                "ATR_VOLATILITY_BREAKOUT",
                atr_breakout["signal"]
            )

        if donchian["signal"] != "NONE":
            return "DONCHIAN_BREAKOUT", donchian["signal"]

    return "NONE", "NONE"


# =========================================================
# 14. STRATEGY ENGINE
# =========================================================

def run_strategy_engine(request):

    highs = request.highs
    lows = request.lows
    closes = request.closes

    validate_candles(highs, lows, closes)

    if any(
        price <= 0
        for price in closes
    ):
        raise HTTPException(
            status_code=400,
            detail="Prices must be greater than zero."
        )

    ema = ema_trend_strategy(closes)

    smc = smc_ict_strategy(
        highs, lows, closes
    )

    pullback = trend_pullback_strategy(
        closes
    )

    donchian = donchian_breakout_strategy(
        highs, lows, closes
    )

    mean_reversion = mean_reversion_strategy(
        closes
    )

    atr_breakout = atr_breakout_strategy(
        highs, lows, closes
    )

    regime_data = detect_market_regime(
        highs, lows, closes
    )

    selected_strategy, selected_signal = select_strategy(
        regime_data["regime"],
        ema,
        smc,
        pullback,
        donchian,
        mean_reversion,
        atr_breakout
    )

    return {
        "symbol": request.symbol.upper(),
        "platform": request.platform,
        "mode": request.mode,

        "market_regime": regime_data,

        "selected_strategy": selected_strategy,
        "signal": selected_signal,

        "strategies": {
            "ema_trend": ema,
            "smc_ict": smc,
            "trend_pullback": pullback,
            "donchian_breakout": donchian,
            "mean_reversion": mean_reversion,
            "atr_breakout": atr_breakout
        },

        "indicators": {
            "rsi": calculate_rsi(closes),
            "atr": regime_data["atr"],
            "adx": regime_data["adx"]
        },

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "message": (
            "Preliminary strategy analysis only. "
            "No order has been placed."
        )
    }


# =========================================================
# 15. BASIC API ENDPOINTS
# =========================================================

@app.get("/")
def home():

    return {
        "name": "Lucidious Nova EA API",
        "version": "2.0.0",
        "status": "online",
        "trading_enabled": TRADING_ENABLED,
        "platforms": SUPPORTED_PLATFORMS,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/api/public-config")
def public_config():

    return {
        "name": "Lucidious Nova EA",
        "version": "2.0.0",
        "supported_markets": STANDARD_MARKETS,
        "platforms": SUPPORTED_PLATFORMS,
        "trading_enabled": False,
        "live_prices_connected": False,
        "automatic_strategy_selection": True,
        "message": "Connect a broker bridge for live market discovery."
    }


@app.get("/api/status")
def status(
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    return {
        "api": "online",
        "mt4_bridge": (
            "connected"
            if BROKER_MARKETS["MT4"]
            else "not_connected"
        ),
        "mt5_bridge": (
            "connected"
            if BROKER_MARKETS["MT5"]
            else "not_connected"
        ),
        "live_trading_enabled": False,
        "message": "Trading remains disabled."
    }


# =========================================================
# 16. TRADING CONTROL
# =========================================================

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
                "Trading is disabled. "
                "A verified execution bridge and safety controls "
                "must be implemented first."
            )
        )

    return {
        "accepted": True,
        "platform": command.platform,
        "action": "stop",
        "trading_enabled": False,
        "message": "Trading remains disabled."
    }


# =========================================================
# 17. AUTOMATIC STRATEGY ENDPOINT
# =========================================================

@app.post("/api/strategy/analyze")
def analyze_market(
    request: StrategyRequest,
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    return run_strategy_engine(request)


# =========================================================
# 18. INDIVIDUAL STRATEGY SIGNAL ENDPOINT
# =========================================================

@app.post("/api/strategy/signal")
def strategy_signal(
    request: StrategyRequest,
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    result = run_strategy_engine(request)

    return {
        "symbol": result["symbol"],
        "platform": result["platform"],
        "signal": result["signal"],
        "strategy": result["selected_strategy"],
        "market_regime": result["market_regime"],
        "message": result["message"]
    }


# =========================================================
# 19. INDIVIDUAL SMC / ICT ENDPOINT
# =========================================================

@app.post("/api/strategy/smc-ict")
def smc_ict_endpoint(
    request: StrategyRequest,
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    validate_candles(
        request.highs,
        request.lows,
        request.closes
    )

    result = smc_ict_strategy(
        request.highs,
        request.lows,
        request.closes
    )

    return {
        "symbol": request.symbol.upper(),
        "platform": request.platform,
        **result
    }


# =========================================================
# 20. BROKER MARKET DISCOVERY
# =========================================================

@app.post("/api/broker/markets")
def receive_broker_markets(
    request: BrokerMarketUpdate,
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    platform = request.platform.upper()

    instruments = [
        item.model_dump()
        for item in request.instruments
    ]

    BROKER_MARKETS[platform] = {
        "broker_name": request.broker_name,
        "platform": platform,
        "account_currency": request.account_currency,
        "instruments": instruments,
        "total_markets": len(instruments),
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    return {
        "success": True,
        "broker": request.broker_name,
        "platform": platform,
        "markets_received": len(instruments),
        "message": "Broker market catalogue received."
    }


# =========================================================
# 21. GET ALL BROKER MARKETS
# =========================================================

@app.get("/api/markets/all")
def get_all_markets(
    platform: str = "MT5",
    category: str | None = None,
    tradable_only: bool = False,
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    platform = platform.upper()

    if platform not in SUPPORTED_PLATFORMS:

        raise HTTPException(
            status_code=400,
            detail="Platform must be MT4 or MT5."
        )

    catalogue = BROKER_MARKETS[platform]

    if not catalogue:

        return {
            "connected": False,
            "platform": platform,
            "markets": [],
            "total": 0,
            "message": "No broker markets received yet."
        }

    markets = catalogue["instruments"]

    if category:

        markets = [
            item for item in markets
            if (
                item.get("category") or ""
            ).lower() == category.lower()
        ]

    if tradable_only:

        markets = [
            item for item in markets
            if item["trade_allowed"]
        ]

    return {
        "connected": True,
        "broker": catalogue["broker_name"],
        "platform": platform,
        "total": len(markets),
        "markets": markets,
        "updated_at": catalogue["updated_at"]
    }


# =========================================================
# 22. SEARCH BROKER MARKETS
# =========================================================

@app.get("/api/markets/search")
def search_markets(
    q: str,
    platform: str = "MT5",
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    platform = platform.upper()

    if platform not in SUPPORTED_PLATFORMS:

        raise HTTPException(
            status_code=400,
            detail="Platform must be MT4 or MT5."
        )

    if not q.strip():

        raise HTTPException(
            status_code=400,
            detail="Enter a market name."
        )

    catalogue = BROKER_MARKETS[platform]

    if not catalogue:

        return {
            "connected": False,
            "results": [],
            "total": 0
        }

    query = q.strip().lower()

    results = [
        item for item in catalogue["instruments"]
        if (
            query in item["name"].lower()
            or query in (
                item.get("description") or ""
            ).lower()
        )
    ]

    return {
        "connected": True,
        "query": q,
        "results": results,
        "total": len(results)
    }


# =========================================================
# 23. MATCH STANDARD MARKET TO BROKER SYMBOL
# =========================================================

@app.get("/api/broker/match")
def match_broker_symbol(
    market: str,
    platform: str = "MT5",
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    platform = platform.upper()
    market = market.upper().strip()

    if platform not in SUPPORTED_PLATFORMS:

        raise HTTPException(
            status_code=400,
            detail="Platform must be MT4 or MT5."
        )

    standard_symbols = {
        symbol
        for group in STANDARD_MARKETS.values()
        for symbol in group
    }

    if market not in standard_symbols:

        raise HTTPException(
            status_code=400,
            detail="Market is not in the standard catalogue."
        )

    catalogue = BROKER_MARKETS[platform]

    if not catalogue:

        return {
            "matched": False,
            "market": market,
            "matches": [],
            "message": "Connect a broker terminal first."
        }

    matches = []

    for item in catalogue["instruments"]:

        name = item["name"].upper()
        base = (
            item.get("base_name") or ""
        ).upper()

        if base == market or name == market:

            matches.append(item)

        elif (
            name.startswith(market)
            and len(name) <= len(market) + 10
        ):

            matches.append(item)

    return {
        "matched": len(matches) > 0,
        "market": market,
        "matches": matches,
        "requires_confirmation": len(matches) > 1,
        "message": (
            "Confirm the correct broker symbol."
            if matches
            else "No matching broker symbol found."
        )
    }


# =========================================================
# 24. STANDARD MARKET CATALOGUE
# =========================================================

@app.get("/api/markets")
def get_supported_markets(
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    return {
        "name": "Lucidious Nova EA",
        "markets": STANDARD_MARKETS,
        "trading_enabled": False,
        "message": "Standard market catalogue."
    }
