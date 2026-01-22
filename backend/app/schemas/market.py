from pydantic import BaseModel
from typing import List, Optional

class LiveDataSchema(BaseModel):
    price: Optional[float]
    change: Optional[float]

class CoinMarketData(BaseModel):
    symbol: str
    price: Optional[float]
    change: Optional[float]

class MarketResponse(BaseModel):
    live_data: dict[str, LiveDataSchema]
    top_gainers: List[CoinMarketData]
    top_losers: List[CoinMarketData]

class CoinSearch(BaseModel):
    name: str
    ticker: str
    slug: str

class PredictionResponse(BaseModel):
    symbol: str
    predicted_price: Optional[str]
    market_cap: str | None
    volume_24h: str | None
    description: str | None
