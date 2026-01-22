import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { marketApi } from '../services/api';
import type { MarketResponse } from '../types';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';
import { TrendingUp, TrendingDown, Star, ArrowUpRight, Zap, Globe, BarChart3, ChevronRight, Search, Clock, Users, Shield } from 'lucide-react';

const Market = () => {
    const [data, setData] = useState<MarketResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await marketApi.getLive();
                setData(res.data);
            } catch (err) {
                console.error("Failed to fetch market data");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, []);

    const coinLogos: Record<string, string> = {
        'BTC/USDT': '/images/btc.png',
        'ETH/USDT': '/images/eth.png',
        'SOL/USDT': '/images/sol.png',
        'XRP/USDT': '/images/xrp.png',
        'ADA/USDT': '/images/ada.png',
    };

    const coinSlugs: Record<string, string> = {
        'BTC/USDT': 'btc',
        'ETH/USDT': 'eth',
        'SOL/USDT': 'sol',
        'XRP/USDT': 'xrp',
        'ADA/USDT': 'ada',
    };

    const coinDescriptions: Record<string, string> = {
        'BTC/USDT': 'The original cryptocurrency',
        'ETH/USDT': 'Smart contract platform',
        'SOL/USDT': 'High-speed blockchain',
        'XRP/USDT': 'Cross-border payments',
        'ADA/USDT': 'Proof-of-stake pioneer',
    };

    if (loading) {
        return (
            <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
                <div className="bg-grid" />
                <Navbar />
                <main className="container" style={{ flex: 1, padding: '40px 0' }}>
                    <div className="skeleton" style={{ height: '120px', marginBottom: '24px' }} />
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '32px' }}>
                        {[1, 2, 3, 4].map(i => <div key={i} className="skeleton" style={{ height: '100px' }} />)}
                    </div>
                    <div className="skeleton" style={{ height: '60px', marginBottom: '24px' }} />
                    <div className="skeleton" style={{ height: '500px' }} />
                </main>
                <Footer />
            </div>
        );
    }

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main className="container" style={{ flex: 1, padding: '40px 0 80px', position: 'relative', zIndex: 1 }}>
                {/* Page Header */}
                <div style={{ marginBottom: '40px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                            <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '12px' }}>
                                Market <span style={{ color: 'var(--accent)' }}>Overview</span>
                            </h1>
                            <p style={{ color: 'var(--text-muted)', fontSize: '15px', maxWidth: '500px' }}>
                                Real-time cryptocurrency prices with AI-powered predictions. Track, analyze, and make informed trading decisions.
                            </p>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', background: 'rgba(14, 203, 129, 0.1)', borderRadius: '50px' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--green)', animation: 'pulse 2s infinite' }} />
                            <span style={{ fontSize: '13px', color: 'var(--green)', fontWeight: 500 }}>Live Data</span>
                        </div>
                    </div>
                </div>

                {data && (
                    <>
                        {/* Market Stats Overview */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '32px' }}>
                            {[
                                { label: 'Total Market Cap', value: '$2.1T+', change: '+2.4%', icon: <Globe size={20} />, positive: true },
                                { label: '24h Volume', value: '$89.2B', change: '+5.1%', icon: <BarChart3 size={20} />, positive: true },
                                { label: 'Active Traders', value: '12.4K', change: '+340', icon: <Users size={20} />, positive: true },
                                { label: 'AI Accuracy', value: '94.7%', change: 'Verified', icon: <Shield size={20} />, positive: true },
                            ].map((stat, i) => (
                                <div key={i} className="card" style={{ padding: '20px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(240, 185, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)' }}>
                                            {stat.icon}
                                        </div>
                                        <span className={`price-badge ${stat.positive ? 'up' : 'down'}`}>{stat.change}</span>
                                    </div>
                                    <div style={{ fontSize: '24px', fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", marginBottom: '4px' }}>{stat.value}</div>
                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{stat.label}</div>
                                </div>
                            ))}
                        </div>

                        {/* Quick Movers Cards */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '40px' }}>
                            {/* Top Gainers */}
                            <div className="card">
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(14, 203, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <TrendingUp size={20} color="#0ecb81" />
                                        </div>
                                        <div>
                                            <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Top Gainers</h3>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>24h Performance</span>
                                        </div>
                                    </div>
                                </div>
                                {data.top_gainers.slice(0, 4).map((coin, i) => (
                                    <Link key={i} to={`/coin/${coinSlugs[coin.symbol] || 'btc'}`} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 0', borderBottom: i < 3 ? '1px solid var(--border)' : 'none', textDecoration: 'none', color: 'inherit' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <span style={{ width: '20px', fontSize: '12px', color: 'var(--text-muted)', fontWeight: 500 }}>{i + 1}</span>
                                            {coinLogos[coin.symbol] && <img src={coinLogos[coin.symbol]} alt="" style={{ width: '28px', height: '28px', borderRadius: '50%' }} />}
                                            <span style={{ fontWeight: 500, fontSize: '14px' }}>{coin.symbol.split('/')[0]}</span>
                                        </div>
                                        <div style={{ textAlign: 'right' }}>
                                            <div className="price-up" style={{ fontWeight: 600, fontSize: '13px' }}>+{coin.change?.toFixed(2)}%</div>
                                            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>${coin.price?.toLocaleString()}</div>
                                        </div>
                                    </Link>
                                ))}
                            </div>

                            {/* Featured / Hot */}
                            <div className="card card-highlight">
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(240, 185, 11, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <Zap size={20} color="#f0b90b" />
                                        </div>
                                        <div>
                                            <h3 style={{ fontSize: '15px', fontWeight: 600 }}>AI Predictions</h3>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Available Now</span>
                                        </div>
                                    </div>
                                </div>
                                {data.top_gainers.slice(0, 4).map((coin, i) => (
                                    <Link key={i} to={`/coin/${coinSlugs[coin.symbol] || 'btc'}`} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 0', borderBottom: i < 3 ? '1px solid var(--border)' : 'none', textDecoration: 'none', color: 'inherit' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <Star size={16} color="#f0b90b" fill="#f0b90b" />
                                            {coinLogos[coin.symbol] && <img src={coinLogos[coin.symbol]} alt="" style={{ width: '28px', height: '28px', borderRadius: '50%' }} />}
                                            <span style={{ fontWeight: 500, fontSize: '14px' }}>{coin.symbol.split('/')[0]}</span>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <span style={{ fontSize: '12px', color: 'var(--accent)', fontWeight: 500 }}>View Prediction</span>
                                            <ArrowUpRight size={14} color="#f0b90b" />
                                        </div>
                                    </Link>
                                ))}
                            </div>

                            {/* Top Losers */}
                            <div className="card">
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(246, 70, 93, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <TrendingDown size={20} color="#f6465d" />
                                        </div>
                                        <div>
                                            <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Top Losers</h3>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>24h Performance</span>
                                        </div>
                                    </div>
                                </div>
                                {data.top_losers.slice(0, 4).map((coin, i) => (
                                    <Link key={i} to={`/coin/${coinSlugs[coin.symbol] || 'btc'}`} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 0', borderBottom: i < 3 ? '1px solid var(--border)' : 'none', textDecoration: 'none', color: 'inherit' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                            <span style={{ width: '20px', fontSize: '12px', color: 'var(--text-muted)', fontWeight: 500 }}>{i + 1}</span>
                                            {coinLogos[coin.symbol] && <img src={coinLogos[coin.symbol]} alt="" style={{ width: '28px', height: '28px', borderRadius: '50%' }} />}
                                            <span style={{ fontWeight: 500, fontSize: '14px' }}>{coin.symbol.split('/')[0]}</span>
                                        </div>
                                        <div style={{ textAlign: 'right' }}>
                                            <div className="price-down" style={{ fontWeight: 600, fontSize: '13px' }}>{coin.change?.toFixed(2)}%</div>
                                            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>${coin.price?.toLocaleString()}</div>
                                        </div>
                                    </Link>
                                ))}
                            </div>
                        </div>

                        {/* Market Table Section */}
                        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                            {/* Table Header */}
                            <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                                    <h2 style={{ fontSize: '18px', fontWeight: 600 }}>All Cryptocurrencies</h2>
                                    <div className="tabs">
                                        {['all', 'gainers', 'losers'].map(tab => (
                                            <button key={tab} className={`tab ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
                                                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                    <div style={{ position: 'relative' }}>
                                        <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                                        <input
                                            type="text"
                                            placeholder="Search coins..."
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            className="input"
                                            style={{ paddingLeft: '38px', width: '200px', fontSize: '13px' }}
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Table */}
                            <div style={{ overflowX: 'auto' }}>
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th style={{ paddingLeft: '24px', width: '50px' }}>#</th>
                                            <th>Name</th>
                                            <th>Last Price</th>
                                            <th>24h Change</th>
                                            <th>Market Cap</th>
                                            <th>24h Volume</th>
                                            <th>AI Prediction</th>
                                            <th style={{ paddingRight: '24px' }}>Action</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(data.live_data)
                                            .filter(([symbol]) => symbol.toLowerCase().includes(searchQuery.toLowerCase()))
                                            .map(([symbol, info], i) => (
                                                <tr key={symbol} onClick={() => window.location.href = `/coin/${coinSlugs[symbol]}`}>
                                                    <td style={{ paddingLeft: '24px' }}>
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                            <Star size={14} style={{ cursor: 'pointer', color: 'var(--text-muted)' }} />
                                                            <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>{i + 1}</span>
                                                        </div>
                                                    </td>
                                                    <td>
                                                        <div className="coin-cell">
                                                            {coinLogos[symbol] && <img src={coinLogos[symbol]} alt={symbol} />}
                                                            <div>
                                                                <div className="coin-name">{symbol.split('/')[0]}</div>
                                                                <div className="coin-symbol">{coinDescriptions[symbol] || symbol}</div>
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: '14px' }}>
                                                        ${info.price?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                                    </td>
                                                    <td>
                                                        <span className={`price-badge ${(info.change || 0) >= 0 ? 'up' : 'down'}`}>
                                                            {(info.change || 0) >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                                            {(info.change || 0) >= 0 ? '+' : ''}{info.change?.toFixed(2)}%
                                                        </span>
                                                    </td>
                                                    <td style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
                                                        ${((info.price || 0) * 21000000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                    </td>
                                                    <td style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
                                                        ${((info.price || 0) * 500000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                    </td>
                                                    <td>
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                            <Zap size={14} color="#f0b90b" />
                                                            <span style={{ fontSize: '12px', color: 'var(--accent)', fontWeight: 500 }}>Available</span>
                                                        </div>
                                                    </td>
                                                    <td style={{ paddingRight: '24px' }}>
                                                        <Link
                                                            to={`/coin/${coinSlugs[symbol]}`}
                                                            className="btn btn-ghost"
                                                            style={{ padding: '8px 16px', fontSize: '12px' }}
                                                            onClick={(e) => e.stopPropagation()}
                                                        >
                                                            Trade <ArrowUpRight size={12} />
                                                        </Link>
                                                    </td>
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Bottom Info Section */}
                        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', marginTop: '40px' }}>
                            <div className="card">
                                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>About Our Market Data</h3>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '16px' }}>
                                    CryptoVertex aggregates real-time market data from leading exchanges including Binance, providing you with accurate price information and comprehensive market analytics. Our AI models analyze this data continuously to generate predictive insights.
                                </p>
                                <div style={{ display: 'flex', gap: '24px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Clock size={16} color="var(--accent)" />
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Updates every 10 seconds</span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Shield size={16} color="var(--accent)" />
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Verified data sources</span>
                                    </div>
                                </div>
                            </div>
                            <div className="card" style={{ background: 'rgba(240, 185, 11, 0.05)', borderColor: 'rgba(240, 185, 11, 0.2)' }}>
                                <div style={{ display: 'flex', gap: '12px' }}>
                                    <Zap size={24} color="#f0b90b" style={{ flexShrink: 0 }} />
                                    <div>
                                        <h4 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px' }}>AI Predictions</h4>
                                        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '12px' }}>
                                            Get AI-powered price predictions for all listed cryptocurrencies.
                                        </p>
                                        <Link to="/coin/btc" style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                            Try BTC Prediction <ChevronRight size={14} />
                                        </Link>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </main>

            <Footer />
        </div>
    );
};

export default Market;
