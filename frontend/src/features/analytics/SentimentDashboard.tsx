import { useEffect, useState } from 'react';
import { AreaChart, Area, Tooltip, ResponsiveContainer } from 'recharts';
import { analyticsApi } from '../../services/api';
import Navbar from '../../components/layout/Navbar';
import Footer from '../../components/layout/Footer';
import { AlertCircle, Twitter, MessageCircle, Zap, MessageSquare } from 'lucide-react';

const SentimentDashboard = () => {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await analyticsApi.getSentiment();
                setData(res.data);
            } catch (err) {
                console.error("Failed to fetch sentiment data");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 30000);
        return () => clearInterval(interval);
    }, []);

    const getGaugeColor = (score: number) => {
        if (score >= 75) return '#0ecb81'; // Extreme Greed
        if (score >= 55) return '#a2cf6e'; // Greed
        if (score >= 45) return '#f0b90b'; // Neutral
        if (score >= 25) return '#f68e5d'; // Fear
        return '#f6465d'; // Extreme Fear
    };

    if (loading) return null; // Or skeleton

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main className="container" style={{ flex: 1, padding: '40px 0', position: 'relative', zIndex: 1 }}>
                <div style={{ marginBottom: '32px' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(240, 185, 11, 0.1)', padding: '6px 14px', borderRadius: '50px', marginBottom: '16px' }}>
                        <Zap size={14} color="#f0b90b" />
                        <span style={{ fontSize: '12px', color: '#f0b90b', fontWeight: 600 }}>AI Market Intelligence</span>
                    </div>
                    <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700 }}>
                        Social <span style={{ color: 'var(--accent)' }}>Sentiment</span>
                    </h1>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>
                        Real-time market emotion analysis from millions of social data points.
                    </p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
                    {/* Gauge Card */}
                    <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px' }}>
                        <div style={{ position: 'relative', width: '200px', height: '100px', marginBottom: '20px' }}>
                            <div style={{
                                width: '200px', height: '100px',
                                background: 'linear-gradient(90deg, #f6465d 0%, #f0b90b 50%, #0ecb81 100%)',
                                borderRadius: '100px 100px 0 0',
                                opacity: 0.2
                            }} />
                            <div style={{
                                position: 'absolute', bottom: 0, left: '50%',
                                width: '4px', height: '90px',
                                background: 'var(--text-primary)',
                                transformOrigin: 'bottom center',
                                transform: `translateX(-50%) rotate(${(data?.score || 50) * 1.8 - 90}deg)`,
                                transition: 'transform 1s ease-out',
                                borderRadius: '4px'
                            }} />
                            <div style={{ position: 'absolute', bottom: -8, left: '50%', transform: 'translateX(-50%)', width: '16px', height: '16px', borderRadius: '50%', background: 'var(--text-primary)' }} />
                        </div>
                        <div style={{ fontSize: '48px', fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif" }}>
                            {data?.score}
                        </div>
                        <div style={{ fontSize: '18px', fontWeight: 600, color: getGaugeColor(data?.score || 50) }}>
                            {data?.label}
                        </div>
                        <div style={{ marginTop: '16px', fontSize: '12px', color: 'var(--text-muted)' }}>
                            Updated 2 min ago
                        </div>
                    </div>

                    {/* Trend Chart */}
                    <div className="card" style={{ padding: '24px' }}>
                        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px' }}>24h Sentiment Trend</h3>
                        <div style={{ height: '220px', width: '100%' }}>
                            <ResponsiveContainer>
                                <AreaChart data={data?.trend}>
                                    <defs>
                                        <linearGradient id="sentimentGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <Tooltip
                                        contentStyle={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: '8px' }}
                                        itemStyle={{ color: 'var(--accent)' }}
                                    />
                                    <Area type="monotone" dataKey="score" stroke="var(--accent)" fill="url(#sentimentGrad)" strokeWidth={2} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* Social Feed */}
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
                    <div className="card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                            <h3 style={{ fontSize: '18px', fontWeight: 600 }}>Live Social Signals</h3>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <span className="badge">Twitter</span>
                                <span className="badge">Reddit</span>
                                <span className="badge">Telegram</span>
                            </div>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                            {data?.posts?.map((post: any, i: number) => (
                                <div key={i} style={{ display: 'flex', gap: '16px', padding: '16px', background: 'var(--bg-primary)', borderRadius: '12px', border: '1px solid var(--border)' }}>
                                    <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        {post.platform === 'Twitter' ? <Twitter size={18} color="#1da1f2" /> :
                                            post.platform === 'Reddit' ? <MessageSquare size={18} color="#ff4500" /> :
                                                <MessageCircle size={18} color="#0088cc" />}
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontWeight: 600, fontSize: '14px' }}>{post.user}</span>
                                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{post.time}</span>
                                        </div>
                                        <p style={{ fontSize: '14px', lineHeight: 1.6, marginBottom: '8px', color: 'var(--text-secondary)' }}>
                                            {post.text}
                                        </p>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                            <span style={{
                                                fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '4px',
                                                background: post.sentiment === 'positive' ? 'rgba(14, 203, 129, 0.1)' : post.sentiment === 'negative' ? 'rgba(246, 70, 93, 0.1)' : 'rgba(240, 185, 11, 0.1)',
                                                color: post.sentiment === 'positive' ? 'var(--green)' : post.sentiment === 'negative' ? 'var(--red)' : 'var(--accent)'
                                            }}>
                                                {post.sentiment.toUpperCase()}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Word Cloud / Trending Topics */}
                    <div className="card">
                        <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '20px' }}>Trending Topics</h3>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
                            {['Bitcoin', 'ETF Approval', 'Bull Run', 'Solana', 'Regulation', 'Fed Rate', 'Binance', 'Memecoins', 'Layer 2', 'Airdrop'].map((topic, i) => (
                                <span key={i} style={{
                                    padding: '8px 16px',
                                    background: i < 3 ? 'var(--accent)' : 'var(--bg-primary)',
                                    color: i < 3 ? '#000' : 'var(--text-primary)',
                                    borderRadius: '50px',
                                    fontSize: i < 3 ? '14px' : '13px',
                                    fontWeight: i < 3 ? 600 : 400,
                                    border: i < 3 ? 'none' : '1px solid var(--border)'
                                }}>
                                    #{topic}
                                </span>
                            ))}
                        </div>

                        <div style={{ marginTop: '32px', padding: '20px', background: 'rgba(240, 185, 11, 0.05)', borderRadius: '12px', border: '1px solid rgba(240, 185, 11, 0.2)' }}>
                            <div style={{ display: 'flex', gap: '12px', marginBottom: '8px' }}>
                                <AlertCircle size={20} color="var(--accent)" />
                                <h4 style={{ fontSize: '14px', fontWeight: 600 }}>Insight</h4>
                            </div>
                            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                                Social volume for Bitcoin has increased by 45% in the last 24h, often preceding high volatility.
                            </p>
                        </div>
                    </div>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default SentimentDashboard;
