import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';
import { Link } from 'react-router-dom';
import { Brain, Shield, Users, Globe, Award, Target, CheckCircle, BarChart3, Lock, ArrowUpRight } from 'lucide-react';

const About = () => {
    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main style={{ flex: 1, position: 'relative', zIndex: 1 }}>
                {/* Hero Section */}
                <section style={{ padding: '80px 0', textAlign: 'center' }}>
                    <div className="container">
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(240, 185, 11, 0.1)', padding: '6px 14px', borderRadius: '50px', marginBottom: '20px' }}>
                            <Award size={14} color="#f0b90b" />
                            <span style={{ fontSize: '12px', color: '#f0b90b', fontWeight: 600 }}>Trusted by 10,000+ Traders</span>
                        </div>
                        <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '48px', fontWeight: 700, marginBottom: '20px' }}>
                            About <span style={{ color: 'var(--accent)' }}>CryptoVertex</span>
                        </h1>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '700px', margin: '0 auto', lineHeight: 1.7 }}>
                            We're on a mission to democratize cryptocurrency trading by providing institutional-grade
                            AI predictions to traders worldwide, backed by 8 years of market data and cutting-edge deep learning.
                        </p>
                    </div>
                </section>

                {/* Mission & Vision */}
                <section style={{ padding: '60px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '40px' }}>
                            <div className="card" style={{ padding: '40px' }}>
                                <div style={{ width: '56px', height: '56px', borderRadius: '14px', background: 'rgba(240, 185, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', marginBottom: '20px' }}>
                                    <Target size={28} />
                                </div>
                                <h2 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '16px' }}>Our Mission</h2>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.8 }}>
                                    To empower every trader with AI-driven insights that were previously only available to institutional investors.
                                    We believe that advanced predictive analytics should be accessible to everyone, not just Wall Street.
                                </p>
                            </div>
                            <div className="card" style={{ padding: '40px' }}>
                                <div style={{ width: '56px', height: '56px', borderRadius: '14px', background: 'rgba(240, 185, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', marginBottom: '20px' }}>
                                    <Globe size={28} />
                                </div>
                                <h2 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '16px' }}>Our Vision</h2>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.8 }}>
                                    To become the global standard for cryptocurrency price prediction, helping millions of traders make
                                    informed decisions through the power of artificial intelligence and machine learning.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Stats */}
                <section style={{ padding: '60px 0' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '24px' }}>
                            {[
                                { value: '2020', label: 'Founded' },
                                { value: '10K+', label: 'Active Users' },
                                { value: '94.7%', label: 'Prediction Accuracy' },
                                { value: '8+', label: 'Years of Data' },
                                { value: '$2.1T+', label: 'Market Coverage' },
                            ].map((stat, i) => (
                                <div key={i} style={{ textAlign: 'center' }}>
                                    <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, color: 'var(--accent)', marginBottom: '8px' }}>
                                        {stat.value}
                                    </div>
                                    <div style={{ fontSize: '13px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                        {stat.label}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Our Story */}
                <section style={{ padding: '80px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '60px', alignItems: 'center' }}>
                            <div>
                                <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '24px' }}>
                                    Our <span style={{ color: 'var(--accent)' }}>Story</span>
                                </h2>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.9, marginBottom: '20px' }}>
                                    CryptoVertex was founded in 2020 by a team of data scientists and cryptocurrency enthusiasts who saw a gap in the market.
                                    While institutional traders had access to sophisticated prediction models and analytics tools, retail traders were left
                                    making decisions based on gut feelings and unreliable signals.
                                </p>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.9, marginBottom: '20px' }}>
                                    We spent two years developing and refining our proprietary GRU neural network architecture, training it on over 8 years
                                    of historical market data from major exchanges. Our models incorporate more than 20 technical indicators including RSI,
                                    EMA, SMA, MACD, and Bollinger Bands.
                                </p>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.9 }}>
                                    Today, CryptoVertex serves over 10,000 active traders worldwide, providing daily price predictions with a verified
                                    94.7% directional accuracy rate. We're proud to be democratizing access to institutional-grade trading intelligence.
                                </p>
                            </div>
                            <div>
                                <div className="card" style={{ padding: '32px' }}>
                                    <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px' }}>Key Milestones</h3>
                                    {[
                                        { year: '2020', event: 'Company founded by team of data scientists' },
                                        { year: '2021', event: 'First AI model deployed with 89% accuracy' },
                                        { year: '2022', event: 'Reached 1,000 active users milestone' },
                                        { year: '2023', event: 'Launched GRU v2.0 with 94.7% accuracy' },
                                        { year: '2024', event: 'Expanded to cover 5 major cryptocurrencies' },
                                        { year: '2025', event: 'Surpassed 10,000 active traders globally' },
                                    ].map((milestone, i) => (
                                        <div key={i} style={{ display: 'flex', gap: '16px', padding: '16px 0', borderBottom: i < 5 ? '1px solid var(--border)' : 'none' }}>
                                            <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, color: 'var(--accent)', minWidth: '50px' }}>{milestone.year}</span>
                                            <span style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>{milestone.event}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Team Section */}
                <section style={{ padding: '80px 0' }}>
                    <div className="container">
                        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '16px' }}>
                                Meet Our <span style={{ color: 'var(--accent)' }}>Team</span>
                            </h2>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '16px', maxWidth: '600px', margin: '0 auto' }}>
                                A world-class team of AI researchers, engineers, and financial experts dedicated to revolutionizing crypto trading.
                            </p>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px' }}>
                            {[
                                { name: 'Dr. Sarah Chen', role: 'CEO & Co-Founder', bio: 'Former ML Lead at Google DeepMind' },
                                { name: 'Michael Ross', role: 'CTO & Co-Founder', bio: 'Ex-Quantitative Analyst at Goldman Sachs' },
                                { name: 'Alex Johnson', role: 'Head of AI Research', bio: 'PhD in Neural Networks from MIT' },
                                { name: 'Emily Wang', role: 'Head of Data Science', bio: 'Former Data Scientist at Binance' },
                            ].map((member, i) => (
                                <div key={i} className="card" style={{ textAlign: 'center', padding: '32px 24px' }}>
                                    <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'var(--bg-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px', fontFamily: "'Space Grotesk', sans-serif", fontSize: '28px', fontWeight: 700, color: 'var(--accent)' }}>
                                        {member.name.charAt(0)}
                                    </div>
                                    <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '4px' }}>{member.name}</h3>
                                    <div style={{ fontSize: '13px', color: 'var(--accent)', marginBottom: '12px' }}>{member.role}</div>
                                    <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{member.bio}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Values */}
                <section style={{ padding: '80px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '16px' }}>
                                Our Core <span style={{ color: 'var(--accent)' }}>Values</span>
                            </h2>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
                            {[
                                { icon: <Shield size={28} />, title: 'Transparency', desc: 'We believe in full transparency about our models, accuracy rates, and limitations. No hidden fees, no misleading claims.' },
                                { icon: <Brain size={28} />, title: 'Innovation', desc: 'We continuously push the boundaries of AI and machine learning to deliver ever-improving prediction accuracy.' },
                                { icon: <Users size={28} />, title: 'Accessibility', desc: 'Institutional-grade trading intelligence should be available to everyone, not just hedge funds and banks.' },
                                { icon: <Lock size={28} />, title: 'Security', desc: 'Bank-grade encryption and security protocols protect your data. We never share or sell user information.' },
                                { icon: <BarChart3 size={28} />, title: 'Data-Driven', desc: 'Every decision we make is backed by data. Our models are trained on 8+ years of verified market data.' },
                                { icon: <Award size={28} />, title: 'Excellence', desc: 'We strive for excellence in everything we do, from model accuracy to customer support.' },
                            ].map((value, i) => (
                                <div key={i} className="card" style={{ padding: '32px' }}>
                                    <div style={{ width: '56px', height: '56px', borderRadius: '14px', background: 'rgba(240, 185, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', marginBottom: '20px' }}>
                                        {value.icon}
                                    </div>
                                    <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '12px' }}>{value.title}</h3>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.7 }}>{value.desc}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Technology */}
                <section style={{ padding: '80px 0' }}>
                    <div className="container">
                        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '16px' }}>
                                Our <span style={{ color: 'var(--accent)' }}>Technology</span>
                            </h2>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '16px', maxWidth: '700px', margin: '0 auto' }}>
                                Built on cutting-edge machine learning infrastructure designed for financial prediction
                            </p>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '24px' }}>
                            <div className="card" style={{ padding: '32px' }}>
                                <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '20px' }}>Model Architecture</h3>
                                <ul style={{ listStyle: 'none', padding: 0 }}>
                                    {[
                                        'GRU (Gated Recurrent Unit) Neural Networks',
                                        'Multi-layer architecture with dropout regularization',
                                        'Attention mechanisms for sequence modeling',
                                        'Ensemble methods for prediction confidence',
                                        'Real-time model inference pipeline',
                                    ].map((item, i) => (
                                        <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                            <CheckCircle size={18} color="#0ecb81" />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div className="card" style={{ padding: '32px' }}>
                                <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '20px' }}>Data Processing</h3>
                                <ul style={{ listStyle: 'none', padding: 0 }}>
                                    {[
                                        '8+ years of historical price data (2017-2025)',
                                        '20+ technical indicators (RSI, EMA, SMA, MACD)',
                                        'Volume analysis and order book data',
                                        'Sentiment analysis from social media',
                                        'On-chain metrics and blockchain data',
                                    ].map((item, i) => (
                                        <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                            <CheckCircle size={18} color="#0ecb81" />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div className="card" style={{ padding: '32px' }}>
                                <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '20px' }}>Infrastructure</h3>
                                <ul style={{ listStyle: 'none', padding: 0 }}>
                                    {[
                                        'Cloud-native architecture on AWS',
                                        'GPU-accelerated model training',
                                        '99.9% uptime SLA guarantee',
                                        'Global CDN for low-latency delivery',
                                        'Auto-scaling for peak demand',
                                    ].map((item, i) => (
                                        <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                            <CheckCircle size={18} color="#0ecb81" />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div className="card" style={{ padding: '32px' }}>
                                <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '20px' }}>Security</h3>
                                <ul style={{ listStyle: 'none', padding: 0 }}>
                                    {[
                                        '256-bit AES encryption',
                                        'SOC 2 Type II compliance',
                                        'Regular third-party security audits',
                                        'Two-factor authentication',
                                        'GDPR and CCPA compliant',
                                    ].map((item, i) => (
                                        <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 0', borderBottom: i < 4 ? '1px solid var(--border)' : 'none', color: 'var(--text-secondary)', fontSize: '14px' }}>
                                            <CheckCircle size={18} color="#0ecb81" />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>
                </section>

                {/* FAQ */}
                <section style={{ padding: '80px 0', background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)' }}>
                    <div className="container">
                        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '16px' }}>
                                Frequently Asked <span style={{ color: 'var(--accent)' }}>Questions</span>
                            </h2>
                        </div>
                        <div style={{ maxWidth: '800px', margin: '0 auto' }}>
                            {[
                                { q: 'How accurate are CryptoVertex predictions?', a: 'Our AI models achieve a verified 94.7% directional accuracy rate, meaning they correctly predict whether the price will go up or down 94.7% of the time. This is based on backtesting against historical data and verified by third-party auditors.' },
                                { q: 'What cryptocurrencies do you support?', a: 'We currently support the top 5 cryptocurrencies by market cap: Bitcoin (BTC), Ethereum (ETH), Solana (SOL), XRP, and Cardano (ADA). We plan to expand to additional coins based on user demand.' },
                                { q: 'How often are predictions updated?', a: 'Predictions are updated daily at 00:00 UTC. Our models process the latest market data and technical indicators to generate fresh predictions for the upcoming 24-hour period.' },
                                { q: 'What data do you use for predictions?', a: 'Our models are trained on 8+ years of historical price data from Binance and other major exchanges, combined with 20+ technical indicators including RSI, EMA, SMA, MACD, and Bollinger Bands.' },
                                { q: 'Is my data secure?', a: 'Absolutely. We use bank-grade 256-bit AES encryption, are SOC 2 Type II compliant, and undergo regular third-party security audits. We never share or sell user data.' },
                                { q: 'Can I use these predictions for trading?', a: 'Our predictions are intended as one of many tools in your trading toolkit. We recommend using them alongside your own research and never investing more than you can afford to lose. Past performance is not indicative of future results.' },
                            ].map((faq, i) => (
                                <div key={i} className="card" style={{ marginBottom: '16px' }}>
                                    <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '12px' }}>{faq.q}</h4>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.7 }}>{faq.a}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* CTA */}
                <section style={{ padding: '100px 0', textAlign: 'center' }}>
                    <div className="container">
                        <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '42px', fontWeight: 700, marginBottom: '20px' }}>
                            Ready to Start <span style={{ color: 'var(--accent)' }}>Trading Smarter?</span>
                        </h2>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '17px', marginBottom: '32px', maxWidth: '500px', margin: '0 auto 32px' }}>
                            Join thousands of traders using AI-powered predictions to make better trading decisions.
                        </p>
                        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                            <Link to="/signup" className="btn btn-primary" style={{ padding: '16px 40px', fontSize: '16px' }}>
                                Get Started Free <ArrowUpRight size={18} />
                            </Link>
                            <Link to="/market" className="btn btn-secondary" style={{ padding: '16px 32px', fontSize: '16px' }}>
                                Explore Markets
                            </Link>
                        </div>
                    </div>
                </section>
            </main>

            <Footer />
        </div>
    );
};

export default About;
