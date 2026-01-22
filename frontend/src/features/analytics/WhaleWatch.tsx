import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { analyticsApi } from '../../services/api';
import Navbar from '../../components/layout/Navbar';
import Footer from '../../components/layout/Footer';
import { Waves, ArrowRight, ArrowDownLeft, ArrowUpRight, Wallet, Activity, AlertCircle } from 'lucide-react';

const WhaleWatch = () => {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await analyticsApi.getWhaleAlerts();
                setData(res.data);
            } catch (err) {
                console.error("Failed to fetch whale data");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 15000);
        return () => clearInterval(interval);
    }, []);

    const formatMoney = (amount: number) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            notation: "compact",
            maximumFractionDigits: 1
        }).format(amount);
    };

    if (loading) return null;

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main className="container" style={{ flex: 1, padding: '40px 0', position: 'relative', zIndex: 1 }}>
                <div style={{ marginBottom: '32px' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(14, 203, 129, 0.1)', padding: '6px 14px', borderRadius: '50px', marginBottom: '16px' }}>
                        <Waves size={14} color="#0ecb81" />
                        <span style={{ fontSize: '12px', color: '#0ecb81', fontWeight: 600 }}>On-Chain Analysis</span>
                    </div>
                    <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700 }}>
                        Whale <span style={{ color: 'var(--accent)' }}>Watcher</span>
                    </h1>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>
                        Track large institutional movements and exchange inflows/outflows in real-time.
                    </p>
                </div>

                {/* Summary Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '32px' }}>
                    <div className="card" style={{ padding: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px', color: 'var(--text-muted)' }}>
                            <ArrowDownLeft size={20} color="#f6465d" />
                            <span>Exchange Inflow (24h)</span>
                        </div>
                        <div style={{ fontSize: '28px', fontWeight: 700 }}>{data?.summary.inflow_24h} BTC</div>
                        <div style={{ fontSize: '12px', color: 'var(--red)', marginTop: '4px' }}>High Selling Pressure</div>
                    </div>
                    <div className="card" style={{ padding: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px', color: 'var(--text-muted)' }}>
                            <ArrowUpRight size={20} color="#0ecb81" />
                            <span>Exchange Outflow (24h)</span>
                        </div>
                        <div style={{ fontSize: '28px', fontWeight: 700 }}>{data?.summary.outflow_24h} BTC</div>
                        <div style={{ fontSize: '12px', color: 'var(--green)', marginTop: '4px' }}>Accumulation Phase</div>
                    </div>
                    <div className="card" style={{ padding: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px', color: 'var(--text-muted)' }}>
                            <Activity size={20} color="var(--accent)" />
                            <span>Net Flow</span>
                        </div>
                        <div style={{ fontSize: '28px', fontWeight: 700, color: data?.summary.net_flow >= 0 ? 'var(--green)' : 'var(--red)' }}>
                            {data?.summary.net_flow > 0 ? '+' : ''}{data?.summary.net_flow} BTC
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>Net movement</div>
                    </div>
                </div>

                {/* Main Content Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>

                    {/* Alert Feed */}
                    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)' }}>
                            <h3 style={{ fontSize: '18px', fontWeight: 600 }}>Large Transactions Feed</h3>
                        </div>
                        <div style={{ overflowX: 'auto' }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th style={{ paddingLeft: '24px' }}>Amount</th>
                                        <th>Value</th>
                                        <th>From</th>
                                        <th>To</th>
                                        <th style={{ paddingRight: '24px' }}>Time</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data?.alerts?.map((alert: any, i: number) => (
                                        <tr key={i}>
                                            <td style={{ paddingLeft: '24px' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                                                    {alert.coin === 'BTC' || alert.coin === 'ETH' ? (
                                                        <span style={{ color: 'var(--accent)' }}>{alert.coin}</span>
                                                    ) : (
                                                        <span>{alert.coin}</span>
                                                    )}
                                                    {alert.amount.toLocaleString()}
                                                </div>
                                            </td>
                                            <td style={{ color: 'var(--text-secondary)' }}>
                                                {formatMoney(alert.value_usd)}
                                            </td>
                                            <td>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                    <Wallet size={14} color="var(--text-muted)" />
                                                    <span style={{ fontSize: '13px' }}>{alert.from_wallet}</span>
                                                </div>
                                            </td>
                                            <td>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                    <ArrowRight size={14} color="var(--text-muted)" />
                                                    <Wallet size={14} color="var(--text-muted)" />
                                                    <span style={{ fontSize: '13px' }}>{alert.to_wallet}</span>
                                                </div>
                                            </td>
                                            <td style={{ paddingRight: '24px', color: 'var(--text-muted)', fontSize: '13px' }}>
                                                {alert.time}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Highlights / Chart */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div className="card" style={{ padding: '24px' }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px' }}>Net Flow Trend</h3>
                            <div style={{ height: '200px' }}>
                                <ResponsiveContainer>
                                    <BarChart data={[
                                        { day: 'Mon', val: -400 }, { day: 'Tue', val: 200 },
                                        { day: 'Wed', val: -150 }, { day: 'Thu', val: 500 },
                                        { day: 'Fri', val: 800 }, { day: 'Sat', val: -200 },
                                        { day: 'Sun', val: 100 }
                                    ]}>
                                        <XAxis dataKey="day" stroke="var(--text-muted)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                                        <Tooltip
                                            contentStyle={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: '8px' }}
                                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        />
                                        <Bar dataKey="val" radius={[4, 4, 4, 4]}>
                                            {[-400, 200, -150, 500, 800, -200, 100].map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry > 0 ? '#0ecb81' : '#f6465d'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="card card-highlight">
                            <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '12px' }}>🚨 Whale Alert</h3>
                            <p style={{ fontSize: '14px', lineHeight: 1.6, marginBottom: '16px' }}>
                                <strong>Unknown Wallet</strong> just transferred <strong>15,000 BTC ($975M)</strong> to <strong>Binance</strong>.
                            </p>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--red)', fontWeight: 600 }}>
                                <AlertCircle size={14} />
                                High Impact Potential
                            </div>
                        </div>
                    </div>

                </div>
            </main>
            <Footer />
        </div>
    );
};

export default WhaleWatch;
