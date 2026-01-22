from fastapi import APIRouter, HTTPException
from backend.app.services.prediction_service import prediction_service
from backend.app.services.market_service import market_service
from backend.app.schemas.market import PredictionResponse

router = APIRouter()

@router.get("/{slug}", response_model=PredictionResponse)
async def get_coin_details(slug: str):
    symbol_ccxt = market_service.symbol_map.get(slug.lower())
    if not symbol_ccxt:
        raise HTTPException(status_code=404, detail="Coin not found")
        
    static_info = market_service.get_coin_static_data(symbol_ccxt)
    prediction = prediction_service.get_prediction(symbol_ccxt)
    
    price_display = f"${prediction:.2f}" if prediction is not None else "Unavailable"
    
    return {
        "symbol": symbol_ccxt,
        "predicted_price": price_display,
        "market_cap": static_info.get("market_cap"),
        "volume_24h": static_info.get("volume"),
        "description": static_info.get("description")
    }

@router.get("/{slug}/historical")
async def get_historical_data(slug: str):
    symbol_ccxt = market_service.symbol_map.get(slug.lower())
    if not symbol_ccxt:
        raise HTTPException(status_code=404, detail="Coin not found")
        
    data = prediction_service.get_historical_data(symbol_ccxt)
    if not data:
        raise HTTPException(status_code=404, detail="Data not available")
    return data
