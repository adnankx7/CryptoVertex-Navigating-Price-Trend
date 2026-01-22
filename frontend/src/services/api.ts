import axios from 'axios';
import type { MarketResponse, PredictionResponse, CoinHistory, PredictionRequest, SignupRequest, AuthResponse } from '../types';

const API_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
});

api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

export const predictionApi = {
    getPrediction: (symbol: string) => api.get<PredictionResponse>(`/predict/${symbol}`),
    trainModel: (request: PredictionRequest) => api.post<any>('/train', request),
    getHistorical: (symbol: string) => api.get<CoinHistory[]>(`/history/${symbol}`),
};

export const marketApi = {
    getLive: () => api.get<MarketResponse>('/market/live'),
};

export const authApi = {
    login: (formData: FormData) => api.post<AuthResponse>('/token', formData),
    signup: (data: SignupRequest) => api.post<any>('/signup', data),
};

export const analyticsApi = {
    getSentiment: () => api.get<any>('/analytics/sentiment'),
    getWhaleAlerts: () => api.get<any>('/analytics/whale-alerts'),
    getPatterns: () => api.get<any>('/analytics/patterns'),
    getCorrelation: () => api.get<any>('/analytics/correlation'),
    getUnlockEvents: () => api.get<any>('/analytics/token-unlocks'),
};

export default api;
