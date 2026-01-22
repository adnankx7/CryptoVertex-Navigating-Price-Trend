import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { marketApi } from '../../services/api';
import type { MarketResponse } from '../../types';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

const MarketOverview = () => {
    const [data, setData] = useState<MarketResponse | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await marketApi.getLive();
                setData(res.data);
            } catch (err) {
                console.error("Failed to fetch market data");
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    if (!data) return <div className="container">Loading market data...</div>;

    return (
        <div className="container">
            <header style={{ marginBottom: '3rem', textAlign: 'center' }}>
                <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem', background: 'linear-gradient(to right, #3b82f6, #06b6d4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                    Real-Time Crypto Intelligence
                </h1>
                <p style={{ color: 'var(--text-secondary)' }}>Advanced price tracking and AI-powered predictions</p>
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginBottom: '3rem' }}>
                {data.top_gainers.slice(0, 3).map((coin) => (
                    <Link to={`/coin/${coin.symbol.split('/')[0].toLowerCase()}`} key={coin.symbol} style={{ textDecoration: 'none', color: 'inherit' }}>
                        <div className="glass-panel">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <TrendingUp color="#10b981" />
                                    <span style={{ fontWeight: 'bold' }}>Top Gainer</span>
                                </div>
                                <span style={{ color: '#10b981', fontWeight: 'bold' }}>+{coin.change?.toFixed(2)}%</span>
                            </div>
                            <h3 style={{ fontSize: '1.5rem', margin: '0 0 0.5rem 0' }}>{coin.symbol}</h3>
                            <div style={{ fontSize: '1.25rem' }}>${coin.price?.toLocaleString()}</div>
                        </div>
                    </Link>
                ))}
                {data.top_losers.slice(0, 3).map((coin) => (
                    <Link to={`/coin/${coin.symbol.split('/')[0].toLowerCase()}`} key={coin.symbol} style={{ textDecoration: 'none', color: 'inherit' }}>
                        <div className="glass-panel">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <TrendingDown color="#ef4444" />
                                    <span style={{ fontWeight: 'bold' }}>Top Loser</span>
                                </div>
                                <span style={{ color: '#ef4444', fontWeight: 'bold' }}>{coin.change?.toFixed(2)}%</span>
                            </div>
                            <h3 style={{ fontSize: '1.5rem', margin: '0 0 0.5rem 0' }}>{coin.symbol}</h3>
                            <div style={{ fontSize: '1.25rem' }}>${coin.price?.toLocaleString()}</div>
                        </div>
                    </Link>
                ))}
            </div>

            <div className="glass-panel">
                <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                    <Activity /> Live Market
                </h2>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                                <th style={{ textAlign: 'left', padding: '1rem' }}>Asset</th>
                                <th style={{ textAlign: 'right', padding: '1rem' }}>Price</th>
                                <th style={{ textAlign: 'right', padding: '1rem' }}>24h Change</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(data.live_data).map(([symbol, info]) => (
                                <tr key={symbol} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                    <td style={{ padding: '1rem' }}>{symbol}</td>
                                    <td style={{ textAlign: 'right', padding: '1rem' }}>${info.price?.toLocaleString()}</td>
                                    <td style={{ textAlign: 'right', padding: '1rem', color: (info.change || 0) >= 0 ? '#10b981' : '#ef4444' }}>
                                        {(info.change || 0) >= 0 ? '+' : ''}{info.change?.toFixed(2)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default MarketOverview;
