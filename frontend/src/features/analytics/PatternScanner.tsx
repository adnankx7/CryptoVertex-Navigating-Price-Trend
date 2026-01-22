import { useEffect, useState } from 'react';
import { analyticsApi } from '../../services/api';
import Navbar from '../../components/layout/Navbar';
import Footer from '../../components/layout/Footer';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface Pattern {
    coin: string;
    pattern: string;
    timeframe: string;
    confidence: number;
    type: string;
    profit_target: number;
}

const PatternScanner = () => {
    const [patterns, setPatterns] = useState<Pattern[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await analyticsApi.getPatterns();
                setPatterns(res.data);
            } catch (err) {
                console.error("Failed to fetch patterns");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main className="container" style={{ flex: 1, padding: '40px 0 80px', position: 'relative', zIndex: 1 }}>
                <div style={{ marginBottom: '40px' }}>
                    <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '12px' }}>
                        Pattern <span style={{ color: 'var(--accent)' }}>Scanner</span>
                    </h1>
                    <p style={{ color: 'var(--text-muted)', fontSize: '15px' }}>
                        AI-powered detection of classical chart patterns across multiple timeframes.
                    </p>
                </div>

                {loading ? (
                    <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>Loading scanner data...</div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
                        {patterns.map((p, index) => (
                            <div key={index} className="glass-panel" style={{ position: 'relative', overflow: 'hidden' }}>
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '4px',
                                    height: '100%',
                                    background: p.type === 'bullish' ? 'var(--green)' : 'var(--red)'
                                }} />

                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                                    <h3 style={{ fontSize: '20px', fontWeight: 700 }}>{p.coin}</h3>
                                    <span style={{
                                        padding: '4px 12px',
                                        borderRadius: '20px',
                                        fontSize: '12px',
                                        fontWeight: 600,
                                        background: p.type === 'bullish' ? 'rgba(14, 203, 129, 0.1)' : 'rgba(246, 70, 93, 0.1)',
                                        color: p.type === 'bullish' ? 'var(--green)' : 'var(--red)'
                                    }}>
                                        {p.type.toUpperCase()}
                                    </span>
                                </div>

                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
                                    {p.type === 'bullish' ? <TrendingUp size={20} color="var(--green)" /> : <TrendingDown size={20} color="var(--red)" />}
                                    <span style={{ fontSize: '16px', fontWeight: 500 }}>{p.pattern}</span>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '14px' }}>
                                    <div style={{ color: 'var(--text-muted)' }}>Timeframe</div>
                                    <div style={{ textAlign: 'right', fontWeight: 600 }}>{p.timeframe}</div>

                                    <div style={{ color: 'var(--text-muted)' }}>Confidence</div>
                                    <div style={{ textAlign: 'right', fontWeight: 600, color: p.confidence > 80 ? 'var(--green)' : 'var(--text-primary)' }}>
                                        {p.confidence}%
                                    </div>

                                    <div style={{ color: 'var(--text-muted)' }}>Target</div>
                                    <div style={{ textAlign: 'right', fontWeight: 600 }}>${p.profit_target.toLocaleString()}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </main>
            <Footer />
        </div>
    );
};

export default PatternScanner;
