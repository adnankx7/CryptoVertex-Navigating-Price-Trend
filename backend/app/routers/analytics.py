from fastapi import APIRouter
import random
from datetime import datetime, timedelta

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/sentiment")
async def get_sentiment():
    # Simulate sentiment score (0-100)
    score = random.randint(45, 75)
    
    # Determine label
    if score >= 75: label = "Extreme Greed"
    elif score >= 55: label = "Greed"
    elif score >= 45: label = "Neutral"
    elif score >= 25: label = "Fear"
    else: label = "Extreme Fear"
    
    # Simulate recent trend
    trend = []
    current = score
    for i in range(24): # Last 24 hours
        change = random.randint(-5, 5)
        current = max(0, min(100, current - change))
        trend.append({"hour": f"{24-i}h ago", "score": current})
    trend.reverse()
    
    # Simulate social posts
    posts = [
        {"platform": "Twitter", "user": "@CryptoKing", "text": "BTC looking bullish above 65k! 🚀 #Bitcoin", "sentiment": "positive", "time": "2m ago"},
        {"platform": "Reddit", "user": "u/trader_joe", "text": "Market seems indecisive today, waiting for CPI data.", "sentiment": "neutral", "time": "15m ago"},
        {"platform": "Telegram", "user": "AlphaSignals", "text": "Whale movement detected on ETH chain.", "sentiment": "positive", "time": "32m ago"},
        {"platform": "Twitter", "user": "@BearWhale", "text": "Sold my bags, improved resistance at 68k.", "sentiment": "negative", "time": "1h ago"},
        {"platform": "Twitter", "user": "@SolanaFan", "text": "SOL ecosystem is exploding right now!", "sentiment": "positive", "time": "1h ago"},
    ]

    return {
        "score": score,
        "label": label,
        "trend": trend,
        "posts": posts
    }

@router.get("/whale-alerts")
async def get_whale_alerts():
    # Simulate large transactions
    coins = ["BTC", "ETH", "SOL", "USDT", "XRP"]
    types = ["Transfer", "Buy", "Sell", "Mint"]
    wallets = ["Binance Hot Wallet", "Coinbase Cold Storage", "Unknown Wallet", "Kraken", "Vitalik.eth"]
    
    alerts = []
    for _ in range(10):
        coin = random.choice(coins)
        amount = random.randint(1000, 100000) if coin in ["BTC", "ETH"] else random.randint(100000, 50000000)
        value = amount * (65000 if coin == "BTC" else 3500 if coin == "ETH" else 150 if coin == "SOL" else 1 if coin == "USDT" else 0.6)
        
        alerts.append({
            "coin": coin,
            "amount": amount,
            "value_usd": value,
            "type": random.choice(types),
            "from_wallet": random.choice(wallets),
            "to_wallet": random.choice(wallets),
            "time": f"{random.randint(1, 59)}m ago"
        })
        
    # Simulate flow summary
    summary = {
        "inflow_24h": random.randint(500, 2000), 
        "outflow_24h": random.randint(400, 1800),
        "net_flow": 0
    }
    summary["net_flow"] = summary["inflow_24h"] - summary["outflow_24h"]

    return {
        "alerts": alerts,
        "summary": summary
    }

@router.get("/patterns")
async def get_patterns():
    patterns = [
        {"coin": "BTC", "pattern": "Bull Flag", "timeframe": "4h", "confidence": 85, "type": "bullish", "profit_target": 68000},
        {"coin": "ETH", "pattern": "Head & Shoulders", "timeframe": "1h", "confidence": 72, "type": "bearish", "profit_target": 3200},
        {"coin": "SOL", "pattern": "Falling Wedge", "timeframe": "1d", "confidence": 91, "type": "bullish", "profit_target": 180},
        {"coin": "XRP", "pattern": "Double Bottom", "timeframe": "4h", "confidence": 65, "type": "bullish", "profit_target": 0.75},
        {"coin": "ADA", "pattern": "Descending Triangle", "timeframe": "4h", "confidence": 60, "type": "bearish", "profit_target": 0.40}
    ]
    return patterns

@router.get("/correlation")
async def get_correlation():
    # Matrix labels must match matrix size
    assets = ["BTC", "ETH", "SOL", "XRP", "SPX", "NDX", "GOLD", "DXY"]
    
    # Simulated correlation matrix (1.0 = perfect correlation, -1.0 = inverse)
    matrix = [
        [1.0, 0.85, 0.62, 0.45, 0.30, 0.35, 0.10, -0.40], # BTC
        [0.85, 1.0, 0.68, 0.50, 0.32, 0.38, 0.12, -0.42], # ETH
        [0.62, 0.68, 1.0, 0.40, 0.25, 0.30, 0.05, -0.35], # SOL
        [0.45, 0.50, 0.40, 1.0, 0.15, 0.20, 0.05, -0.20], # XRP
        [0.30, 0.32, 0.25, 0.15, 1.0, 0.92, 0.10, -0.50], # SPX
        [0.35, 0.38, 0.30, 0.20, 0.92, 1.0, 0.08, -0.55], # NDX
        [0.10, 0.12, 0.05, 0.05, 0.10, 0.08, 1.0, -0.60], # GOLD
        [-0.40, -0.42, -0.35, -0.20, -0.50, -0.55, -0.60, 1.0] # DXY
    ]
    return {"assets": assets, "matrix": matrix}

@router.get("/token-unlocks")
async def get_token_unlocks():
    unlocks = [
        {"project": "Arbitrum", "token": "ARB", "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"), "amount": "1.1B", "value": "$850M", "percent_supply": 8.7},
        {"project": "Sui", "token": "SUI", "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"), "amount": "34.6M", "value": "$42M", "percent_supply": 2.4},
        {"project": "Aptos", "token": "APT", "date": (datetime.now() + timedelta(days=12)).strftime("%Y-%m-%d"), "amount": "24.8M", "value": "$210M", "percent_supply": 6.2},
        {"project": "Optimism", "token": "OP", "date": (datetime.now() + timedelta(days=18)).strftime("%Y-%m-%d"), "amount": "24.1M", "value": "$45M", "percent_supply": 2.1},
        {"project": "dYdX", "token": "DYDX", "date": (datetime.now() + timedelta(days=25)).strftime("%Y-%m-%d"), "amount": "2.16M", "value": "$5.2M", "percent_supply": 0.8}
    ]
    return unlocks
