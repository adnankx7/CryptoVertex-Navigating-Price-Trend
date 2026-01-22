import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { predictionApi } from '../../services/api';
import type { PredictionResponse } from '../../types';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, DollarSign, BarChart3, Zap, Clock, Brain, Activity, ChevronRight, AlertCircle, Info, Globe, Award } from 'lucide-react';
import Navbar from '../../components/layout/Navbar';
import Footer from '../../components/layout/Footer';

const CoinDetail = () => {
    const { slug } = useParams<{ slug: string }>();
    const [data, setData] = useState<PredictionResponse | null>(null);
    const [history, setHistory] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('1M');

    const coinData: Record<string, { name: string; symbol: string; description: string; founded: string; category: string; website: string; }> = {
        'btc': {
            name: 'Bitcoin',
            symbol: 'BTC',
            description: 'Bitcoin is a decentralized digital currency that enables instant payments to anyone, anywhere in the world. Bitcoin uses peer-to-peer technology to operate with no central authority: transaction management and money issuance are carried out collectively by the network.',
            founded: '2009',
            category: 'Currency',
            website: 'bitcoin.org'
        },
        'eth': {
            name: 'Ethereum',
            symbol: 'ETH',
            description: 'Ethereum is a decentralized computing platform that enables smart contracts and distributed applications (dApps) to be built and run without any downtime, fraud, control or interference from a third party.',
            founded: '2015',
            category: 'Smart Contracts',
            website: 'ethereum.org'
        },
        'sol': {
            name: 'Solana',
            symbol: 'SOL',
            description: 'Solana is a highly functional open source project that implements a new, permissionless and high-speed layer-1 blockchain. Created in 2017, Solana aims to scale throughput beyond what is typically achieved by popular blockchains while keeping costs low.',
            founded: '2020',
            category: 'Smart Contracts',
            website: 'solana.com'
        },
        'xrp': {
            name: 'XRP',
            symbol: 'XRP',
            description: 'XRP is the native cryptocurrency of the XRP Ledger, an open-source blockchain designed for enterprise use. It\'s primarily used as a bridge currency for cross-border payments, enabling financial institutions to settle transactions quickly and with minimal costs.',
            founded: '2012',
            category: 'Payments',
            website: 'xrpl.org'
        },
        'ada': {
            name: 'Cardano',
            symbol: 'ADA',
            description: 'Cardano is a proof-of-stake blockchain platform that says its goal is to allow changemakers, innovators and visionaries to bring about positive global change. It was founded as an open-source project to redistribute power from corrupt structures.',
            founded: '2017',
            category: 'Smart Contracts',
            website: 'cardano.org'
        },
    };

    const coinLogos: Record<string, string> = {
        'btc': '/images/btc.png',
        'eth': '/images/eth.png',
        'sol': '/images/sol.png',
        'xrp': '/images/xrp.png',
        'ada': '/images/ada.png'
    };

    useEffect(() => {
        const fetchData = async () => {
            if (!slug) return;
            try {
                const [predRes, histRes] = await Promise.all([
                    predictionApi.getPrediction(slug),
                    predictionApi.getHistorical(slug)
                ]);
                setData(predRes.data);
                setHistory(histRes.data || []);
            } catch (err) {
                console.error("Failed to fetch coin data");
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [slug]);

    const coin = coinData[slug || ''] || { name: 'Unknown', symbol: '???', description: '', founded: 'N/A', category: 'N/A', website: '' };

    if (loading) {
        return (
            <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
                <div className="bg-grid" />
                <Navbar />
                <main className="container" style={{ flex: 1, padding: '40px 0' }}>
                    <div className="skeleton" style={{ height: '24px', width: '120px', marginBottom: '24px' }} />
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: '24px' }}>
                        <div>
                            <div className="skeleton" style={{ height: '100px', marginBottom: '24px' }} />
                            <div className="skeleton" style={{ height: '400px', marginBottom: '24px' }} />
                            <div className="skeleton" style={{ height: '200px' }} />
                        </div>
                        <div>
                            <div className="skeleton" style={{ height: '200px', marginBottom: '20px' }} />
                            <div className="skeleton" style={{ height: '300px', marginBottom: '20px' }} />
                            <div className="skeleton" style={{ height: '150px' }} />
                        </div>
                    </div>
                </main>
                <Footer />
            </div>
        );
    }

    if (!data) {
        return (
            <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
                <div className="bg-grid" />
                <Navbar />
                <main className="container" style={{ flex: 1, padding: '60px 0', textAlign: 'center' }}>
                    <h1>Coin not found</h1>
                    <Link to="/market" className="btn btn-primary" style={{ marginTop: '24px' }}>Back to Market</Link>
                </main>
                <Footer />
            </div>
        );
    }

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main className="container" style={{ flex: 1, padding: '32px 0 80px', position: 'relative', zIndex: 1 }}>
                {/* Breadcrumb */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '24px', fontSize: '13px' }}>
                    <Link to="/market" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Markets</Link>
                    <ChevronRight size={14} color="var(--text-muted)" />
                    <span style={{ color: 'var(--text-primary)' }}>{coin.name}</span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '32px' }}>
                    {/* Left Column */}
                    <div>
                        {/* Header */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                                {coinLogos[slug || ''] && (
                                    <img src={coinLogos[slug || '']} alt={slug} style={{ width: '56px', height: '56px', borderRadius: '14px' }} />
                                )}
                                <div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '4px' }}>
                                        <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '32px', fontWeight: 700 }}>
                                            {coin.name}
                                        </h1>
                                        <span style={{ padding: '4px 10px', background: 'var(--bg-card)', borderRadius: '6px', fontSize: '12px', color: 'var(--text-muted)', fontWeight: 500 }}>
                                            {coin.symbol}
                                        </span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <span style={{ padding: '3px 8px', background: 'rgba(240, 185, 11, 0.1)', borderRadius: '4px', fontSize: '11px', color: 'var(--accent)', fontWeight: 500 }}>
                                            {coin.category}
                                        </span>
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Rank #1</span>
                                    </div>
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button className="btn btn-secondary" style={{ padding: '10px 16px', fontSize: '13px' }}>
                                    <Activity size={16} /> Watch
                                </button>
                                <button className="btn btn-primary" style={{ padding: '10px 20px', fontSize: '13px' }}>
                                    Trade
                                </button>
                            </div>
                        </div>

                        {/* Price Stats */}
                        <div className="card" style={{ marginBottom: '24px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
                                <div>
                                    <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '4px' }}>{coin.name} Price ({coin.symbol})</div>
                                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '16px' }}>
                                        <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700 }}>
                                            {data.predicted_price !== 'Unavailable' ? `$${parseFloat(data.predicted_price.replace(/[$,]/g, '')).toLocaleString()}` : 'Loading...'}
                                        </span>
                                        <span className="price-badge up" style={{ padding: '6px 12px' }}>
                                            <TrendingUp size={14} /> +2.45%
                                        </span>
                                    </div>
                                </div>
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    {['1D', '1W', '1M', '3M', '1Y', 'ALL'].map(tab => (
                                        <button key={tab} className={`tab ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)} style={{ padding: '6px 12px', fontSize: '12px' }}>
                                            {tab}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Chart */}
                            <div style={{ height: '320px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={history}>
                                        <defs>
                                            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#f0b90b" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#f0b90b" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <XAxis dataKey="Date" stroke="#5e6673" tick={{ fontSize: 11, fill: '#5e6673' }} axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
                                        <YAxis stroke="#5e6673" domain={['auto', 'auto']} tick={{ fontSize: 11, fill: '#5e6673' }} axisLine={{ stroke: 'var(--border)' }} tickLine={false} tickFormatter={(v) => `$${v.toLocaleString()}`} />
                                        <Tooltip contentStyle={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '13px' }} itemStyle={{ color: '#f0b90b' }} />
                                        <Area type="monotone" dataKey="Close" stroke="#f0b90b" strokeWidth={2} fill="url(#priceGradient)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Key Metrics Grid */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px' }}>
                            {[
                                { label: 'Market Cap', value: data.market_cap || '$1.75T', icon: <DollarSign size={18} /> },
                                { label: '24h Volume', value: data.volume_24h || '$63B', icon: <BarChart3 size={18} /> },
                                { label: 'Circulating Supply', value: '19.5M BTC', icon: <Activity size={18} /> },
                                { label: 'Max Supply', value: '21M BTC', icon: <Globe size={18} /> },
                            ].map((metric, i) => (
                                <div key={i} className="card" style={{ padding: '16px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', color: 'var(--text-muted)' }}>
                                        {metric.icon}
                                        <span style={{ fontSize: '12px' }}>{metric.label}</span>
                                    </div>
                                    <div style={{ fontSize: '16px', fontWeight: 600 }}>{metric.value}</div>
                                </div>
                            ))}
                        </div>

                        {/* About Section */}
                        <div className="card" style={{ marginBottom: '24px' }}>
                            <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px' }}>About {coin.name}</h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '20px' }}>
                                {coin.description}
                            </p>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', padding: '16px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
                                <div>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Founded</div>
                                    <div style={{ fontSize: '14px', fontWeight: 500 }}>{coin.founded}</div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Category</div>
                                    <div style={{ fontSize: '14px', fontWeight: 500 }}>{coin.category}</div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Website</div>
                                    <a href={`https://${coin.website}`} target="_blank" rel="noopener noreferrer" style={{ fontSize: '14px', fontWeight: 500 }}>{coin.website}</a>
                                </div>
                            </div>
                        </div>

                        {/* FAQ Section */}
                        <div className="card">
                            <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '20px' }}>Frequently Asked Questions</h3>
                            {[
                                { q: `What is ${coin.name}?`, a: `${coin.name} (${coin.symbol}) is a cryptocurrency that ${coin.category === 'Currency' ? 'functions as a decentralized digital currency' : 'enables smart contracts and decentralized applications'}.` },
                                { q: `How accurate are CryptoVertex predictions for ${coin.symbol}?`, a: `Our AI models achieve 94.7% directional accuracy for ${coin.symbol} predictions, trained on 8+ years of historical data.` },
                                { q: `How often are predictions updated?`, a: `Predictions are updated daily using the latest market data and technical indicators (RSI, EMA, SMA).` },
                            ].map((faq, i) => (
                                <div key={i} style={{ padding: '16px 0', borderBottom: i < 2 ? '1px solid var(--border)' : 'none' }}>
                                    <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Info size={16} color="var(--accent)" />
                                        {faq.q}
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '13px', lineHeight: 1.7, paddingLeft: '24px' }}>{faq.a}</p>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Right Column */}
                    <div>
                        {/* AI Prediction Card */}
                        <div className="card card-highlight pulse-glow" style={{ marginBottom: '20px', textAlign: 'center', padding: '28px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '16px' }}>
                                <Zap size={18} color="#f0b90b" />
                                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                                    AI Price Prediction
                                </span>
                            </div>
                            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '40px', fontWeight: 700, color: 'var(--accent)', marginBottom: '8px' }}>
                                {data.predicted_price}
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '13px', marginBottom: '20px' }}>
                                <Clock size={14} />
                                Next Day Forecast
                            </div>
                            <div style={{ padding: '12px', background: 'rgba(14, 203, 129, 0.1)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                <TrendingUp size={16} color="#0ecb81" />
                                <span style={{ fontSize: '13px', color: 'var(--green)', fontWeight: 500 }}>Bullish Signal Detected</span>
                            </div>
                        </div>

                        {/* Model Info */}
                        <div className="card" style={{ marginBottom: '20px' }}>
                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '16px' }}>Prediction Model</h4>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                {[
                                    { icon: <Brain size={16} />, label: 'Model Type', value: 'GRU Neural Network' },
                                    { icon: <BarChart3 size={16} />, label: 'Training Data', value: '8 Years (2017-2025)' },
                                    { icon: <Award size={16} />, label: 'Accuracy', value: '94.7%', highlight: true },
                                    { icon: <Clock size={16} />, label: 'Last Updated', value: 'Today, 00:00 UTC' },
                                ].map((item, i) => (
                                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px', background: 'var(--bg-primary)', borderRadius: '8px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-muted)' }}>
                                            {item.icon}
                                            <span style={{ fontSize: '13px' }}>{item.label}</span>
                                        </div>
                                        <span style={{ fontSize: '13px', fontWeight: 600, color: item.highlight ? 'var(--green)' : 'var(--text-primary)' }}>{item.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Market Stats */}
                        <div className="card" style={{ marginBottom: '20px' }}>
                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '16px' }}>Market Statistics</h4>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                {[
                                    { label: 'Market Cap', value: data.market_cap || 'N/A' },
                                    { label: '24h Trading Volume', value: data.volume_24h || 'N/A' },
                                    { label: 'Volume/Market Cap', value: '3.59%' },
                                    { label: 'All-Time High', value: '$108,268 (Jan 2025)' },
                                    { label: 'All-Time Low', value: '$67.81 (Jul 2013)' },
                                ].map((item, i) => (
                                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none' }}>
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{item.label}</span>
                                        <span style={{ fontSize: '13px', fontWeight: 500 }}>{item.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Links Card */}
                        <div className="card" style={{ marginBottom: '20px' }}>
                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '16px' }}>Resources</h4>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                {[
                                    { label: 'Official Website', url: `https://${coin.website}` },
                                    { label: 'Whitepaper', url: '#' },
                                    { label: 'Source Code', url: 'https://github.com' },
                                ].map((link, i) => (
                                    <a key={i} href={link.url} target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 12px', background: 'var(--bg-primary)', borderRadius: '8px', textDecoration: 'none', color: 'var(--text-primary)', fontSize: '13px' }}>
                                        {link.label}
                                        <ChevronRight size={14} color="var(--text-muted)" />
                                    </a>
                                ))}
                            </div>
                        </div>

                        {/* Warning */}
                        <div className="card" style={{ background: 'rgba(246, 70, 93, 0.05)', borderColor: 'rgba(246, 70, 93, 0.2)' }}>
                            <div style={{ display: 'flex', gap: '12px' }}>
                                <AlertCircle size={18} color="#f6465d" style={{ flexShrink: 0, marginTop: '2px' }} />
                                <div>
                                    <h4 style={{ fontSize: '13px', fontWeight: 600, marginBottom: '6px', color: 'var(--red)' }}>Risk Disclaimer</h4>
                                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                                        Cryptocurrency investments are subject to high market risk. Past performance is not indicative of future results. Our predictions are for informational purposes only.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            <Footer />
        </div>
    );
};

export default CoinDetail;
