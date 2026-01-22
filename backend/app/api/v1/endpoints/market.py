from fastapi import APIRouter
from backend.app.services.market_service import market_service
from backend.app.schemas.market import MarketResponse, CoinSearch

router = APIRouter()

@router.on_event("startup")
async def start_market_services():
    market_service.start_background_tasks()

@router.get("/live", response_model=MarketResponse)
async def get_live_data():
    return market_service.get_market_summary()

@router.get("/search", response_model=list[CoinSearch])
async def search_coins():
    search_data = []
    for slug, ccxt_symbol in market_service.symbol_map.items():
        coin_info = market_service.get_coin_static_data(ccxt_symbol)
        if coin_info:
            search_data.append({
                'name': coin_info.get('full_name'),
                'ticker': coin_info.get('ticker_symbol'),
                'slug': slug
            })
    return search_data
