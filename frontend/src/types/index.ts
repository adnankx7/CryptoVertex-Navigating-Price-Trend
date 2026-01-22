export interface User {
    username: string;
    email: string;
}

export interface LiveData {
    price: number | null;
    change: number | null;
}

export interface CoinMarketData {
    symbol: string;
    price: number | null;
    change: number | null;
}

export interface MarketResponse {
    live_data: Record<string, LiveData>;
    top_gainers: CoinMarketData[];
    top_losers: CoinMarketData[];
}

export interface PredictionResponse {
    symbol: string;
    predicted_price: string;
    market_cap: string | null;
    volume_24h: string | null;
    description: string | null;
}

export interface CoinSearch {
    name: string;
    ticker: string;
    slug: string;
}

export interface CoinHistory {
    date: string;
    price: number;
}

export interface PredictionRequest {
    symbol: string;
    data_source?: string;
}

export interface SignupRequest {
    username: string;
    email: string;
    password: string;
}

export interface AuthResponse {
    access_token: string;
    token_type: string;
}
