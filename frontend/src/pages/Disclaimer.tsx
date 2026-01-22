import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';

import { AlertTriangle, Shield, Info, FileText, Scale, Ban, AlertCircle, CheckCircle, HelpCircle } from 'lucide-react';

const Disclaimer = () => {
    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main style={{ flex: 1, position: 'relative', zIndex: 1 }}>
                {/* Hero */}
                <section style={{ padding: '80px 0', textAlign: 'center' }}>
                    <div className="container">
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(246, 70, 93, 0.1)', padding: '6px 14px', borderRadius: '50px', marginBottom: '20px' }}>
                            <AlertTriangle size={14} color="#f6465d" />
                            <span style={{ fontSize: '12px', color: '#f6465d', fontWeight: 600 }}>Important Information</span>
                        </div>
                        <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '48px', fontWeight: 700, marginBottom: '20px' }}>
                            Legal <span style={{ color: 'var(--accent)' }}>Disclaimer</span>
                        </h1>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '700px', margin: '0 auto', lineHeight: 1.7 }}>
                            Please read this disclaimer carefully before using our platform. By accessing CryptoVertex,
                            you acknowledge that you have read, understood, and agree to be bound by these terms.
                        </p>
                        <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '16px' }}>
                            Last updated: January 2025
                        </p>
                    </div>
                </section>

                {/* Quick Summary */}
                <section style={{ padding: '40px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px' }}>
                            {[
                                { icon: <AlertCircle size={24} />, title: 'Not Financial Advice', desc: 'Our predictions are for informational purposes only' },
                                { icon: <Scale size={24} />, title: 'High Risk', desc: 'Cryptocurrency trading involves substantial risk of loss' },
                                { icon: <Ban size={24} />, title: 'No Guarantees', desc: 'Past performance does not guarantee future results' },
                                { icon: <Shield size={24} />, title: 'Your Responsibility', desc: 'You are solely responsible for your trading decisions' },
                            ].map((item, i) => (
                                <div key={i} className="card" style={{ textAlign: 'center', padding: '24px' }}>
                                    <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(246, 70, 93, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--red)', margin: '0 auto 16px' }}>
                                        {item.icon}
                                    </div>
                                    <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px' }}>{item.title}</h3>
                                    <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{item.desc}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Main Content */}
                <section style={{ padding: '60px 0' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '40px' }}>
                            {/* Sidebar Navigation */}
                            <div>
                                <div className="card" style={{ position: 'sticky', top: '100px', padding: '20px' }}>
                                    <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '16px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Contents</h4>
                                    <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                        {[
                                            'General Disclaimer',
                                            'Risk Warnings',
                                            'Prediction Accuracy',
                                            'No Financial Advice',
                                            'Investment Risks',
                                            'Limitation of Liability',
                                            'Third-Party Content',
                                            'Regulatory Compliance',
                                            'User Responsibilities',
                                            'Contact Information',
                                        ].map((item, i) => (
                                            <a key={i} href={`#section-${i}`} style={{ fontSize: '13px', color: 'var(--text-secondary)', textDecoration: 'none', padding: '8px 12px', borderRadius: '6px', transition: 'all 0.15s' }}>
                                                {item}
                                            </a>
                                        ))}
                                    </nav>
                                </div>
                            </div>

                            {/* Content */}
                            <div>
                                {/* General Disclaimer */}
                                <div id="section-0" className="card" style={{ marginBottom: '24px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <FileText size={24} color="var(--accent)" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>1. General Disclaimer</h2>
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '16px' }}>
                                        CryptoVertex ("we," "us," or "our") provides cryptocurrency price predictions and market analysis through our website and services.
                                        The information provided on this platform is for general informational and educational purposes only and should not be construed
                                        as financial, investment, trading, or any other type of professional advice.
                                    </p>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '16px' }}>
                                        We make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability,
                                        suitability, or availability of the information, predictions, or related graphics contained on this platform for any purpose.
                                    </p>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8 }}>
                                        Any reliance you place on such information is therefore strictly at your own risk. In no event will we be liable for any loss
                                        or damage including without limitation, indirect or consequential loss or damage, arising from or in connection with the use of this platform.
                                    </p>
                                </div>

                                {/* Risk Warnings */}
                                <div id="section-1" className="card" style={{ marginBottom: '24px', background: 'rgba(246, 70, 93, 0.05)', borderColor: 'rgba(246, 70, 93, 0.2)' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <AlertTriangle size={24} color="#f6465d" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>2. Risk Warnings</h2>
                                    </div>
                                    <div style={{ background: 'rgba(246, 70, 93, 0.1)', padding: '20px', borderRadius: '10px', marginBottom: '20px', border: '1px solid rgba(246, 70, 93, 0.3)' }}>
                                        <p style={{ color: 'var(--red)', fontSize: '15px', fontWeight: 600, marginBottom: '12px' }}>
                                            ⚠️ HIGH RISK WARNING: Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor.
                                        </p>
                                        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8 }}>
                                            The valuation of cryptocurrencies may fluctuate wildly and, as a result, you may lose more than your original investment.
                                            Do not invest money you cannot afford to lose.
                                        </p>
                                    </div>
                                    <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Key Risk Factors:</h4>
                                    <ul style={{ listStyle: 'none', padding: 0 }}>
                                        {[
                                            'Extreme price volatility – cryptocurrencies can lose 50% or more of their value in a single day',
                                            'Regulatory uncertainty – governments may implement new regulations or bans at any time',
                                            'Market manipulation – cryptocurrency markets are susceptible to pump-and-dump schemes',
                                            'Technical risks – smart contract vulnerabilities, exchange hacks, and wallet breaches',
                                            'Liquidity risks – some cryptocurrencies may become illiquid, making it difficult to sell',
                                            'Counterparty risks – exchanges and platforms may fail or become insolvent',
                                        ].map((risk, i) => (
                                            <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', padding: '12px 0', borderBottom: i < 5 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                                <AlertCircle size={16} color="#f6465d" style={{ flexShrink: 0, marginTop: '2px' }} />
                                                {risk}
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Prediction Accuracy */}
                                <div id="section-2" className="card" style={{ marginBottom: '24px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <Info size={24} color="var(--accent)" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>3. Prediction Accuracy</h2>
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '16px' }}>
                                        Our AI models have achieved a verified 94.7% directional accuracy rate based on historical backtesting. However, this accuracy
                                        rate should be interpreted with the following caveats:
                                    </p>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '20px' }}>
                                        <div style={{ padding: '16px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
                                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', color: 'var(--green)' }}>What 94.7% Means</h4>
                                            <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Our models correctly predict the direction of price movement (up or down) 94.7% of the time based on historical testing.</p>
                                        </div>
                                        <div style={{ padding: '16px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
                                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', color: 'var(--red)' }}>What It Doesn't Mean</h4>
                                            <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>This does not guarantee future results, exact price targets, or the magnitude of price movements.</p>
                                        </div>
                                    </div>
                                    <ul style={{ listStyle: 'none', padding: 0 }}>
                                        {[
                                            'Past performance is not indicative of future results',
                                            'Accuracy rates are based on historical backtesting and may not reflect real-world trading conditions',
                                            'Market conditions change, and our models may not perform as well during unprecedented events',
                                            'Predictions are updated daily and may change based on new market data',
                                        ].map((item, i) => (
                                            <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '10px 0', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                                <HelpCircle size={16} color="var(--accent)" />
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {/* No Financial Advice */}
                                <div id="section-3" className="card" style={{ marginBottom: '24px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <Ban size={24} color="var(--accent)" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>4. Not Financial Advice</h2>
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '16px' }}>
                                        The content on CryptoVertex, including but not limited to price predictions, market analysis, and educational materials,
                                        is not intended to be and does not constitute:
                                    </p>
                                    <ul style={{ listStyle: 'none', padding: 0, marginBottom: '20px' }}>
                                        {[
                                            'Financial advice',
                                            'Investment advice',
                                            'Trading advice',
                                            'Tax advice',
                                            'Legal advice',
                                            'A recommendation to buy, sell, or hold any cryptocurrency',
                                        ].map((item, i) => (
                                            <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '10px 0', borderBottom: i < 5 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                                <Ban size={16} color="#f6465d" />
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8 }}>
                                        Before making any investment decisions, you should seek advice from a qualified financial advisor who is aware of your
                                        specific circumstances and can provide personalized guidance.
                                    </p>
                                </div>

                                {/* Investment Risks */}
                                <div id="section-4" className="card" style={{ marginBottom: '24px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <Scale size={24} color="var(--accent)" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>5. Investment Risks</h2>
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '20px' }}>
                                        Investing in cryptocurrencies carries significant risks that you should understand before making any investment:
                                    </p>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                                        {[
                                            { title: 'Volatility Risk', desc: 'Cryptocurrency prices can change dramatically within hours or even minutes.' },
                                            { title: 'Regulatory Risk', desc: 'Government actions or new laws could significantly impact cryptocurrency values.' },
                                            { title: 'Technology Risk', desc: 'Blockchain technology is still evolving and may have undiscovered vulnerabilities.' },
                                            { title: 'Liquidity Risk', desc: 'You may not be able to sell your assets quickly at a fair price.' },
                                            { title: 'Security Risk', desc: 'Cryptocurrencies are targets for hackers and may be lost through theft or fraud.' },
                                            { title: 'Loss of Capital', desc: 'You could lose some or all of your invested capital.' },
                                        ].map((risk, i) => (
                                            <div key={i} style={{ padding: '16px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
                                                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px' }}>{risk.title}</h4>
                                                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{risk.desc}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Limitation of Liability */}
                                <div id="section-5" className="card" style={{ marginBottom: '24px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <Shield size={24} color="var(--accent)" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>6. Limitation of Liability</h2>
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '16px' }}>
                                        To the maximum extent permitted by applicable law, CryptoVertex and its affiliates, officers, directors, employees,
                                        agents, and licensors will not be liable for:
                                    </p>
                                    <ul style={{ listStyle: 'none', padding: 0 }}>
                                        {[
                                            'Any indirect, incidental, special, consequential, or punitive damages',
                                            'Any loss of profits, revenues, data, or business opportunities',
                                            'Any losses arising from trading decisions based on our predictions',
                                            'Any damages arising from unauthorized access to or use of our services',
                                            'Any interruption or cessation of transmission to or from our services',
                                        ].map((item, i) => (
                                            <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', padding: '10px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                                <CheckCircle size={16} color="var(--accent)" style={{ flexShrink: 0, marginTop: '2px' }} />
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {/* User Responsibilities */}
                                <div id="section-8" className="card" style={{ marginBottom: '24px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <CheckCircle size={24} color="var(--accent)" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>7. User Responsibilities</h2>
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '20px' }}>
                                        By using CryptoVertex, you acknowledge and agree that:
                                    </p>
                                    <ul style={{ listStyle: 'none', padding: 0 }}>
                                        {[
                                            'You are solely responsible for your own investment decisions',
                                            'You should conduct your own research before making any investment',
                                            'You should only invest money that you can afford to lose',
                                            'You should consult with a qualified financial advisor before investing',
                                            'You will comply with all applicable laws and regulations in your jurisdiction',
                                            'You will not rely solely on our predictions for trading decisions',
                                        ].map((item, i) => (
                                            <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 0', borderBottom: i < 5 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                                <CheckCircle size={16} color="#0ecb81" />
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Contact */}
                                <div id="section-9" className="card">
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                                        <Info size={24} color="var(--accent)" />
                                        <h2 style={{ fontSize: '22px', fontWeight: 700 }}>8. Contact Information</h2>
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.8, marginBottom: '20px' }}>
                                        If you have any questions about this disclaimer or our services, please contact us:
                                    </p>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                                        <div style={{ padding: '16px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
                                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px' }}>Email</h4>
                                            <a href="mailto:legal@cryptovertex.com" style={{ fontSize: '14px' }}>legal@cryptovertex.com</a>
                                        </div>
                                        <div style={{ padding: '16px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
                                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px' }}>Support</h4>
                                            <a href="mailto:support@cryptovertex.com" style={{ fontSize: '14px' }}>support@cryptovertex.com</a>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Final Warning */}
                <section style={{ padding: '60px 0', background: 'rgba(246, 70, 93, 0.05)', borderTop: '1px solid rgba(246, 70, 93, 0.2)' }}>
                    <div className="container" style={{ textAlign: 'center' }}>
                        <AlertTriangle size={48} color="#f6465d" style={{ marginBottom: '20px' }} />
                        <h2 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '16px' }}>Final Risk Warning</h2>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '16px', maxWidth: '700px', margin: '0 auto', lineHeight: 1.8 }}>
                            Trading cryptocurrencies is highly speculative and may result in the loss of your entire investment.
                            Do not trade with money you cannot afford to lose. If you are unsure about the suitability of cryptocurrency
                            trading for your circumstances, seek independent financial advice.
                        </p>
                    </div>
                </section>
            </main>

            <Footer />
        </div>
    );
};

export default Disclaimer;
