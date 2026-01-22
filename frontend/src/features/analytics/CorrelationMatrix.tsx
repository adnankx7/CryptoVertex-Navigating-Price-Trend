import { useEffect, useState } from 'react';
import { analyticsApi } from '../../services/api';
import Navbar from '../../components/layout/Navbar';
import Footer from '../../components/layout/Footer';

interface CorrelationData {
    assets: string[];
    matrix: number[][];
}

const CorrelationMatrix = () => {
    const [data, setData] = useState<CorrelationData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await analyticsApi.getCorrelation();
                setData(res.data);
            } catch (err) {
                console.error("Failed to fetch correlation data");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const getColor = (value: number) => {
        // Red for inverse correlation (-1), Green for positive (+1), Grey for neutral (0)
        if (value >= 0) {
            // Green scale
            return `rgba(14, 203, 129, ${value})`; // increasing opacity with correlation
        } else {
            // Red scale
            return `rgba(246, 70, 93, ${Math.abs(value)})`;
        }
    };

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main className="container" style={{ flex: 1, padding: '40px 0 80px', position: 'relative', zIndex: 1 }}>
                <div style={{ marginBottom: '40px' }}>
                    <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '12px' }}>
                        Correlation <span style={{ color: 'var(--accent)' }}>Matrix</span>
                    </h1>
                    <p style={{ color: 'var(--text-muted)', fontSize: '15px' }}>
                        Analyze how different assets move in relation to each other. Values range from -1 (Inverse) to +1 (Perfect).
                    </p>
                </div>

                {loading || !data ? (
                    <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>Loading matrix...</div>
                ) : (
                    <div className="glass-panel" style={{ overflowX: 'auto' }}>
                        <table style={{ borderCollapse: 'collapse', width: '100%', minWidth: '600px' }}>
                            <thead>
                                <tr>
                                    <th style={{ padding: '12px' }}></th>
                                    {data.assets.map(asset => (
                                        <th key={asset} style={{ padding: '12px', color: 'var(--text-muted)' }}>{asset}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {data.matrix.map((row, rowIndex) => (
                                    <tr key={rowIndex}>
                                        <td style={{ padding: '12px', fontWeight: 600 }}>{data.assets[rowIndex]}</td>
                                        {row.map((value, colIndex) => (
                                            <td key={colIndex} style={{ padding: '4px' }}>
                                                <div style={{
                                                    background: getColor(value),
                                                    color: Math.abs(value) > 0.5 ? '#fff' : 'var(--text-primary)',
                                                    padding: '12px',
                                                    borderRadius: '8px',
                                                    textAlign: 'center',
                                                    fontWeight: 600,
                                                    fontSize: '14px',
                                                    border: value === 1 ? '1px solid rgba(255,255,255,0.2)' : 'none'
                                                }}>
                                                    {value.toFixed(2)}
                                                </div>
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </main>
            <Footer />
        </div>
    );
};

export default CorrelationMatrix;
