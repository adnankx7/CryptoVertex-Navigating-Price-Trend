import { useEffect, useState } from 'react';
import { analyticsApi } from '../../services/api';
import Navbar from '../../components/layout/Navbar';
import Footer from '../../components/layout/Footer';
import { Lock, Unlock, Calendar, DollarSign, PieChart } from 'lucide-react';

interface UnlockEvent {
    project: string;
    token: string;
    date: string;
    amount: string;
    value: string;
    percent_supply: number;
}

const TokenUnlocks = () => {
    const [unlocks, setUnlocks] = useState<UnlockEvent[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await analyticsApi.getUnlockEvents();
                setUnlocks(res.data);
            } catch (err) {
                console.error("Failed to fetch unlock events");
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
                        Token <span style={{ color: 'var(--accent)' }}>Unlocks</span>
                    </h1>
                    <p style={{ color: 'var(--text-muted)', fontSize: '15px' }}>
                        Monitor upcoming token vesting events to anticipate potential market impact.
                    </p>
                </div>

                {loading ? (
                    <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>Loading calendar...</div>
                ) : (
                    <div style={{ display: 'grid', gap: '24px' }}>
                        {unlocks.map((e, index) => (
                            <div key={index} className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', minWidth: '200px' }}>
                                    <div style={{
                                        width: '48px',
                                        height: '48px',
                                        background: 'rgba(59, 130, 246, 0.1)',
                                        borderRadius: '12px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        color: 'var(--accent)'
                                    }}>
                                        <Lock size={24} />
                                    </div>
                                    <div>
                                        <h3 style={{ fontSize: '18px', fontWeight: 700, margin: '0 0 4px 0' }}>{e.project}</h3>
                                        <span style={{ color: 'var(--text-muted)', fontSize: '14px' }}>{e.token}</span>
                                    </div>
                                </div>

                                <div style={{ display: 'flex', gap: '40px', flex: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '13px' }}>
                                            <Calendar size={14} /> Date
                                        </div>
                                        <div style={{ fontWeight: 600 }}>{e.date}</div>
                                    </div>

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '13px' }}>
                                            <Unlock size={14} /> Amount
                                        </div>
                                        <div style={{ fontWeight: 600 }}>{e.amount} {e.token}</div>
                                    </div>

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '13px' }}>
                                            <DollarSign size={14} /> Value
                                        </div>
                                        <div style={{ fontWeight: 600 }}>{e.value}</div>
                                    </div>

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '13px' }}>
                                            <PieChart size={14} /> % of Supply
                                        </div>
                                        <div style={{ fontWeight: 600, color: 'var(--accent)' }}>{e.percent_supply}%</div>
                                    </div>
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

export default TokenUnlocks;
