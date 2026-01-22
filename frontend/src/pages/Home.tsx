import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { marketApi } from '../services/api';
import type { MarketResponse } from '../types';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';
import { TrendingUp, TrendingDown, Zap, Shield, BarChart3, Brain, ChevronRight, ArrowUpRight, Lock, Globe, Users, Award, CheckCircle, Star, Clock } from 'lucide-react';

const Home = () => {
    const [marketData, setMarketData] = useState<MarketResponse | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await marketApi.getLive();
                setMarketData(res.data);
            } catch (err) {
                console.error("Failed to fetch market data");
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

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <div className="bg-glow" />
            <Navbar />

            {/* Live Price Ticker */}
            {marketData && (
                <div className="ticker-bar">
                    <div className="container">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '48px', overflow: 'hidden' }}>
                            {Object.entries(marketData.live_data).map(([symbol, data]) => (
                                <Link key={symbol} to={`/coin/${coinSlugs[symbol]}`} className="ticker-item" style={{ textDecoration: 'none', color: 'inherit' }}>
                                    <div className="ticker-coin">
                                        {coinLogos[symbol] && <img src={coinLogos[symbol]} alt={symbol} />}
                                        <span>{symbol.split('/')[0]}</span>
                                    </div>
                                    <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600 }}>
                                        ${data.price?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                    </span>
                                    <span className={`price-badge ${(data.change || 0) >= 0 ? 'up' : 'down'}`}>
                                        {(data.change || 0) >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                        {(data.change || 0) >= 0 ? '+' : ''}{data.change?.toFixed(2)}%
                                    </span>
                                </Link>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            <main style={{ flex: 1 }}>
                {/* Hero Section */}
                <section style={{ padding: '80px 0 60px' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '60px', alignItems: 'center' }}>
                            {/* Left Content */}
                            <div className="fade-in">
                                <div style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    background: 'rgba(240, 185, 11, 0.1)',
                                    padding: '6px 14px',
                                    borderRadius: '50px',
                                    marginBottom: '20px',
                                    border: '1px solid rgba(240, 185, 11, 0.2)',
                                }}>
                                    <Zap size={14} color="#f0b90b" />
                                    <span style={{ fontSize: '12px', color: '#f0b90b', fontWeight: 600 }}>
                                        Trusted by 10,000+ Traders Worldwide
                                    </span>
                                </div>

                                <h1 style={{
                                    fontFamily: "'Space Grotesk', sans-serif",
                                    fontSize: '52px',
                                    fontWeight: 800,
                                    lineHeight: 1.1,
                                    marginBottom: '20px',
                                }}>
                                    The Future of
                                    <br />
                                    <span style={{ color: 'var(--accent)' }}>Crypto Trading</span>
                                </h1>

                                <p style={{
                                    fontSize: '17px',
                                    color: 'var(--text-secondary)',
                                    marginBottom: '28px',
                                    maxWidth: '480px',
                                    lineHeight: 1.7,
                                }}>
                                    Our institutional-grade AI analyzes 8 years of market data to deliver
                                    accurate daily price predictions with 94.7% directional accuracy.
                                </p>

                                <div style={{ display: 'flex', gap: '12px', marginBottom: '32px' }}>
                                    <Link to="/signup" className="btn btn-primary" style={{ padding: '14px 32px', fontSize: '15px' }}>
                                        Start Free Trial
                                        <ArrowUpRight size={16} />
                                    </Link>
                                    <Link to="/market" className="btn btn-secondary" style={{ padding: '14px 28px', fontSize: '15px' }}>
                                        Explore Markets
                                    </Link>
                                </div>

                                {/* Trust Indicators */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '24px', paddingTop: '20px', borderTop: '1px solid var(--border)' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Shield size={18} color="#0ecb81" />
                                        <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Bank-grade Security</span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Lock size={18} color="#0ecb81" />
                                        <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>256-bit Encryption</span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Globe size={18} color="#0ecb81" />
                                        <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Global Access</span>
                                    </div>
                                </div>
                            </div>

                            {/* Right - Market Preview Card */}
                            <div className="fade-in" style={{ animationDelay: '0.1s' }}>
                                <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
                                    <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div>
                                            <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>Live Market</h3>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Real-time prices</span>
                                        </div>
                                        <Link to="/market" style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                            View all <ChevronRight size={14} />
                                        </Link>
                                    </div>

                                    <div style={{ padding: '0 24px' }}>
                                        {marketData && Object.entries(marketData.live_data).slice(0, 5).map(([symbol, data], i) => (
                                            <Link
                                                key={symbol}
                                                to={`/coin/${coinSlugs[symbol]}`}
                                                style={{
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'space-between',
                                                    padding: '16px 0',
                                                    borderBottom: i < 4 ? '1px solid var(--border)' : 'none',
                                                    textDecoration: 'none',
                                                    color: 'inherit',
                                                }}
                                            >
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                                    {coinLogos[symbol] && (
                                                        <img src={coinLogos[symbol]} alt={symbol} style={{ width: '40px', height: '40px', borderRadius: '50%' }} />
                                                    )}
                                                    <div>
                                                        <div style={{ fontWeight: 600, fontSize: '14px' }}>{symbol.split('/')[0]}</div>
                                                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{symbol}</div>
                                                    </div>
                                                </div>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: '14px' }}>
                                                        ${data.price?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                                    </div>
                                                    <div className={`price-badge ${(data.change || 0) >= 0 ? 'up' : 'down'}`} style={{ marginTop: '4px' }}>
                                                        {(data.change || 0) >= 0 ? '+' : ''}{data.change?.toFixed(2)}%
                                                    </div>
                                                </div>
                                            </Link>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Stats Section */}
                <section style={{ padding: '48px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '24px' }}>
                            {[
                                { value: '$2.1T+', label: 'Market Coverage', icon: <DollarSign size={20} /> },
                                { value: '94.7%', label: 'Prediction Accuracy', icon: <TrendingUp size={20} /> },
                                { value: '8+', label: 'Years of Data', icon: <Clock size={20} /> },
                                { value: '10K+', label: 'Active Traders', icon: <Users size={20} /> },
                                { value: '99.9%', label: 'Uptime SLA', icon: <Award size={20} /> },
                            ].map((stat, i) => (
                                <div key={i} style={{ textAlign: 'center' }}>
                                    <div style={{
                                        width: '48px', height: '48px', borderRadius: '12px',
                                        background: 'rgba(240, 185, 11, 0.1)', display: 'flex',
                                        alignItems: 'center', justifyContent: 'center',
                                        color: 'var(--accent)', margin: '0 auto 12px',
                                    }}>
                                        {stat.icon}
                                    </div>
                                    <div style={{
                                        fontFamily: "'Space Grotesk', sans-serif",
                                        fontSize: '32px', fontWeight: 700, color: 'var(--text-primary)',
                                        marginBottom: '4px',
                                    }}>
                                        {stat.value}
                                    </div>
                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                        {stat.label}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* How It Works Section */}
                <section style={{ padding: '80px 0' }}>
                    <div className="container">
                        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '16px' }}>
                                How <span style={{ color: 'var(--accent)' }}>CryptoVertex</span> Works
                            </h2>
                            <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto', fontSize: '15px' }}>
                                Our AI-powered platform combines deep learning with technical analysis to deliver accurate cryptocurrency price predictions
                            </p>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '32px' }}>
                            {[
                                {
                                    step: '01',
                                    title: 'Data Collection',
                                    desc: 'We aggregate 8 years of historical price data, volume, and market indicators from Binance and other major exchanges.',
                                    icon: <BarChart3 size={28} />
                                },
                                {
                                    step: '02',
                                    title: 'AI Analysis',
                                    desc: 'Our GRU neural networks process RSI, EMA, SMA, and 20+ technical indicators to identify patterns and trends.',
                                    icon: <Brain size={28} />
                                },
                                {
                                    step: '03',
                                    title: 'Price Prediction',
                                    desc: 'Get accurate daily price forecasts with confidence intervals to make informed trading decisions.',
                                    icon: <Zap size={28} />
                                },
                            ].map((item, i) => (
                                <div key={i} className="card" style={{ position: 'relative', padding: '32px' }}>
                                    <div style={{
                                        position: 'absolute', top: '24px', right: '24px',
                                        fontFamily: "'Space Grotesk', sans-serif",
                                        fontSize: '48px', fontWeight: 700,
                                        color: 'rgba(240, 185, 11, 0.1)',
                                    }}>
                                        {item.step}
                                    </div>
                                    <div style={{
                                        width: '56px', height: '56px', borderRadius: '14px',
                                        background: 'rgba(240, 185, 11, 0.1)', display: 'flex',
                                        alignItems: 'center', justifyContent: 'center',
                                        color: 'var(--accent)', marginBottom: '20px',
                                    }}>
                                        {item.icon}
                                    </div>
                                    <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '12px' }}>{item.title}</h3>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.7 }}>{item.desc}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* AI Model Showcase */}
                <section style={{ padding: '80px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '60px', alignItems: 'center' }}>
                            <div>
                                <div style={{
                                    display: 'inline-flex', alignItems: 'center', gap: '8px',
                                    background: 'rgba(240, 185, 11, 0.1)', padding: '6px 12px',
                                    borderRadius: '50px', marginBottom: '20px',
                                }}>
                                    <Brain size={14} color="#f0b90b" />
                                    <span style={{ fontSize: '12px', color: '#f0b90b', fontWeight: 600 }}>Deep Learning</span>
                                </div>
                                <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '20px' }}>
                                    Institutional-Grade <span style={{ color: 'var(--accent)' }}>AI Models</span>
                                </h2>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.8, marginBottom: '28px' }}>
                                    Our proprietary GRU neural network architecture processes multi-dimensional market data
                                    to generate highly accurate price predictions across major cryptocurrencies.
                                </p>
                                <ul style={{ listStyle: 'none', padding: 0, marginBottom: '28px' }}>
                                    {[
                                        'Trained on 8+ years of historical market data',
                                        'Incorporates 20+ technical indicators',
                                        'Real-time data processing pipeline',
                                        'Continuous model retraining and optimization',
                                    ].map((item, i) => (
                                        <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '14px', color: 'var(--text-primary)', fontSize: '14px' }}>
                                            <CheckCircle size={18} color="#0ecb81" />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                                <Link to="/coin/btc" className="btn btn-primary">
                                    View BTC Prediction <ArrowUpRight size={16} />
                                </Link>
                            </div>
                            <div className="card" style={{ padding: '24px' }}>
                                <img
                                    src="/images/gru-arch.png"
                                    alt="Neural Network Architecture"
                                    style={{ width: '100%', borderRadius: '12px' }}
                                />
                                <div style={{ marginTop: '16px', padding: '16px', background: 'var(--bg-primary)', borderRadius: '8px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Model Type</span>
                                        <span style={{ fontSize: '13px', fontWeight: 600 }}>GRU Neural Network</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Training Data</span>
                                        <span style={{ fontSize: '13px', fontWeight: 600 }}>8 Years (2017-2025)</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Accuracy</span>
                                        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--green)' }}>94.7%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Top Movers */}
                {marketData && (
                    <section style={{ padding: '80px 0' }}>
                        <div className="container">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
                                <div>
                                    <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '28px', fontWeight: 700, marginBottom: '8px' }}>
                                        Market <span style={{ color: 'var(--accent)' }}>Movers</span>
                                    </h2>
                                    <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Real-time performance across our supported cryptocurrencies</p>
                                </div>
                                <Link to="/market" className="btn btn-ghost">
                                    View All Markets <ChevronRight size={16} />
                                </Link>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
                                {/* Top Gainers */}
                                <div className="card">
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(14, 203, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <TrendingUp size={20} color="#0ecb81" />
                                        </div>
                                        <div>
                                            <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Top Gainers</h3>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>24h Performance</span>
                                        </div>
                                    </div>
                                    {marketData.top_gainers.slice(0, 4).map((coin, i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '14px 0', borderBottom: i < 3 ? '1px solid var(--border)' : 'none' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <span style={{ width: '20px', fontSize: '12px', color: 'var(--text-muted)' }}>{i + 1}</span>
                                                <span style={{ fontWeight: 500, fontSize: '14px' }}>{coin.symbol}</span>
                                            </div>
                                            <span className="price-up" style={{ fontWeight: 600, fontSize: '13px' }}>+{coin.change?.toFixed(2)}%</span>
                                        </div>
                                    ))}
                                </div>

                                {/* Featured */}
                                <div className="card card-highlight">
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(240, 185, 11, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <Star size={20} color="#f0b90b" fill="#f0b90b" />
                                        </div>
                                        <div>
                                            <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Featured Coins</h3>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>AI Predictions Active</span>
                                        </div>
                                    </div>
                                    {marketData.top_gainers.slice(0, 4).map((coin, i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 0', borderBottom: i < 3 ? '1px solid var(--border)' : 'none' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <Zap size={14} color="#f0b90b" />
                                                <span style={{ fontWeight: 500, fontSize: '14px' }}>{coin.symbol}</span>
                                            </div>
                                            <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>${coin.price?.toLocaleString()}</span>
                                        </div>
                                    ))}
                                </div>

                                {/* Top Losers */}
                                <div className="card">
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(246, 70, 93, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <TrendingDown size={20} color="#f6465d" />
                                        </div>
                                        <div>
                                            <h3 style={{ fontSize: '15px', fontWeight: 600 }}>Top Losers</h3>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>24h Performance</span>
                                        </div>
                                    </div>
                                    {marketData.top_losers.slice(0, 4).map((coin, i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '14px 0', borderBottom: i < 3 ? '1px solid var(--border)' : 'none' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <span style={{ width: '20px', fontSize: '12px', color: 'var(--text-muted)' }}>{i + 1}</span>
                                                <span style={{ fontWeight: 500, fontSize: '14px' }}>{coin.symbol}</span>
                                            </div>
                                            <span className="price-down" style={{ fontWeight: 600, fontSize: '13px' }}>{coin.change?.toFixed(2)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </section>
                )}

                {/* Testimonials */}
                <section style={{ padding: '80px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '32px', fontWeight: 700, marginBottom: '12px' }}>
                                Trusted by <span style={{ color: 'var(--accent)' }}>Traders</span> Worldwide
                            </h2>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '15px' }}>See what our users are saying about CryptoVertex</p>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
                            {[
                                { name: 'Alex Chen', role: 'Day Trader', text: 'CryptoVertex predictions have become an essential part of my trading strategy. The accuracy is remarkable.' },
                                { name: 'Sarah Johnson', role: 'Portfolio Manager', text: 'Finally, an AI platform that delivers on its promises. The 8 years of data really makes a difference.' },
                                { name: 'Michael Ross', role: 'Crypto Investor', text: 'The best investment decision I made was subscribing to CryptoVertex. It pays for itself many times over.' },
                            ].map((testimonial, i) => (
                                <div key={i} className="card">
                                    <div style={{ display: 'flex', gap: '4px', marginBottom: '16px' }}>
                                        {[1, 2, 3, 4, 5].map(star => <Star key={star} size={16} color="#f0b90b" fill="#f0b90b" />)}
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.7, marginBottom: '20px' }}>
                                        "{testimonial.text}"
                                    </p>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--bg-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, color: 'var(--accent)' }}>
                                            {testimonial.name.charAt(0)}
                                        </div>
                                        <div>
                                            <div style={{ fontWeight: 600, fontSize: '14px' }}>{testimonial.name}</div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{testimonial.role}</div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* CTA Section */}
                <section style={{ padding: '100px 0', position: 'relative', overflow: 'hidden' }}>
                    <div style={{
                        position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                        width: '800px', height: '400px',
                        background: 'radial-gradient(ellipse, rgba(240, 185, 11, 0.1) 0%, transparent 60%)',
                        pointerEvents: 'none',
                    }} />
                    <div className="container" style={{ textAlign: 'center', position: 'relative' }}>
                        <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '42px', fontWeight: 700, marginBottom: '20px' }}>
                            Start Trading with <span style={{ color: 'var(--accent)' }}>AI Insights</span>
                        </h2>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '17px', marginBottom: '36px', maxWidth: '500px', margin: '0 auto 36px' }}>
                            Join thousands of traders using CryptoVertex to make data-driven decisions
                        </p>
                        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                            <Link to="/signup" className="btn btn-primary" style={{ padding: '16px 40px', fontSize: '16px' }}>
                                Get Started Free
                                <ArrowUpRight size={18} />
                            </Link>
                            <Link to="/about" className="btn btn-secondary" style={{ padding: '16px 32px', fontSize: '16px' }}>
                                Learn More
                            </Link>
                        </div>
                    </div>
                </section>
            </main>

            <Footer />
        </div>
    );
};

// Helper icon component
const DollarSign = ({ size }: { size: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="1" x2="12" y2="23"></line>
        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
    </svg>
);

export default Home;
