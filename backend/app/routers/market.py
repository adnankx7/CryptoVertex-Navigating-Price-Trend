from fastapi import APIRouter
import random

router = APIRouter(prefix="/market", tags=["market"])

@router.get("/live")
async def get_live_market_data():
    # Base prices for simulation
    base_prices = {
        "BTC/USDT": 64230.50,
        "ETH/USDT": 3450.20,
        "SOL/USDT": 145.80,
        "XRP/USDT": 0.62,
        "ADA/USDT": 0.45,
        "DOGE/USDT": 0.16,
        "DOT/USDT": 7.20,
        "MATIC/USDT": 0.75,
        "LINK/USDT": 14.50,
        "UNI/USDT": 7.80
    }
    
    live_data = {}
    top_gainers = []
    top_losers = []
    
    # Generate simulated data for each coin
    for symbol, base_price in base_prices.items():
        # Random fluctuation within 2%
        change_pct = random.uniform(-5.0, 5.0)
        current_price = base_price * (1 + change_pct / 100)
        
        coin_data = {
            "symbol": symbol,
            "price": current_price,
            "change": change_pct
        }
        
        live_data[symbol] = {
            "price": current_price,
            "change": change_pct
        }
        
        if change_pct > 0:
            top_gainers.append(coin_data)
        else:
            top_losers.append(coin_data)
            
    # Sort gainers and losers
    top_gainers.sort(key=lambda x: x["change"], reverse=True)
    top_losers.sort(key=lambda x: x["change"])
    
    return {
        "live_data": live_data,
        "top_gainers": top_gainers,
        "top_losers": top_losers
    }
