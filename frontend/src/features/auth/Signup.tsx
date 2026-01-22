import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { authApi } from '../../services/api';
import Navbar from '../../components/layout/Navbar';

import { Mail, Lock, User, Shield, Zap, CheckCircle, ArrowRight, Gift, BarChart3, Globe, Clock } from 'lucide-react';

const Signup = () => {
    const navigate = useNavigate();
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        setLoading(true);
        setError('');

        try {
            await authApi.signup({ username, email, password });
            navigate('/login');
        } catch {
            setError('Failed to create account. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main style={{ flex: 1, display: 'flex', position: 'relative', zIndex: 1 }}>
                {/* Left Side - Form */}
                <div style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '60px',
                }}>
                    <div style={{ width: '100%', maxWidth: '440px' }}>
                        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
                            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(14, 203, 129, 0.1)', padding: '6px 14px', borderRadius: '50px', marginBottom: '16px' }}>
                                <Gift size={14} color="#0ecb81" />
                                <span style={{ fontSize: '12px', color: '#0ecb81', fontWeight: 600 }}>Start Free - No Credit Card Required</span>
                            </div>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '28px', fontWeight: 700, marginBottom: '8px' }}>
                                Create Your Account
                            </h2>
                            <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                                Join 10,000+ traders using AI-powered predictions
                            </p>
                        </div>

                        <div className="card" style={{ padding: '32px' }}>
                            <form onSubmit={handleSubmit}>
                                {error && (
                                    <div style={{ background: 'rgba(246, 70, 93, 0.1)', border: '1px solid rgba(246, 70, 93, 0.3)', borderRadius: '10px', padding: '12px 16px', marginBottom: '20px', color: 'var(--red)', fontSize: '13px' }}>
                                        {error}
                                    </div>
                                )}

                                <div style={{ marginBottom: '16px' }}>
                                    <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>
                                        Username
                                    </label>
                                    <div style={{ position: 'relative' }}>
                                        <User size={18} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                                        <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} className="input" placeholder="johndoe" style={{ paddingLeft: '44px' }} required />
                                    </div>
                                </div>

                                <div style={{ marginBottom: '16px' }}>
                                    <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>
                                        Email Address
                                    </label>
                                    <div style={{ position: 'relative' }}>
                                        <Mail size={18} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                                        <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="input" placeholder="you@example.com" style={{ paddingLeft: '44px' }} required />
                                    </div>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '20px' }}>
                                    <div>
                                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>
                                            Password
                                        </label>
                                        <div style={{ position: 'relative' }}>
                                            <Lock size={18} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                                            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="input" placeholder="••••••••" style={{ paddingLeft: '44px' }} required minLength={6} />
                                        </div>
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>
                                            Confirm
                                        </label>
                                        <div style={{ position: 'relative' }}>
                                            <Lock size={18} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                                            <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="input" placeholder="••••••••" style={{ paddingLeft: '44px' }} required />
                                        </div>
                                    </div>
                                </div>

                                <div style={{ marginBottom: '24px' }}>
                                    <label style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', cursor: 'pointer' }}>
                                        <input type="checkbox" required style={{ marginTop: '4px' }} />
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                            I agree to the <Link to="/disclaimer" style={{ fontWeight: 500 }}>Terms of Service</Link> and <Link to="/disclaimer" style={{ fontWeight: 500 }}>Privacy Policy</Link>
                                        </span>
                                    </label>
                                </div>

                                <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: '100%', justifyContent: 'center', padding: '14px', fontSize: '15px' }}>
                                    {loading ? 'Creating account...' : <>Create Free Account <ArrowRight size={16} /></>}
                                </button>

                                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', margin: '24px 0' }}>
                                    <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>or sign up with</span>
                                    <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                                    <button type="button" className="btn btn-secondary" style={{ padding: '12px' }}>
                                        <svg width="18" height="18" viewBox="0 0 24 24"><path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" /><path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" /><path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" /><path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" /></svg>
                                        Google
                                    </button>
                                    <button type="button" className="btn btn-secondary" style={{ padding: '12px' }}>
                                        <svg width="18" height="18" viewBox="0 0 24 24"><path fill="currentColor" d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /></svg>
                                        GitHub
                                    </button>
                                </div>
                            </form>
                        </div>

                        <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '14px', color: 'var(--text-muted)' }}>
                            Already have an account?{' '}
                            <Link to="/login" style={{ fontWeight: 600 }}>Sign in</Link>
                        </p>
                    </div>
                </div>

                {/* Right Side - Benefits */}
                <div style={{
                    flex: 1,
                    background: 'var(--bg-secondary)',
                    borderLeft: '1px solid var(--border)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    padding: '60px',
                }}>
                    <div style={{ maxWidth: '480px' }}>
                        <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '32px', fontWeight: 700, marginBottom: '24px' }}>
                            What You'll Get with <span style={{ color: 'var(--accent)' }}>CryptoVertex</span>
                        </h2>

                        {/* Benefits */}
                        <div style={{ marginBottom: '40px' }}>
                            {[
                                { icon: <Zap size={20} />, title: 'AI Price Predictions', desc: 'Daily forecasts for BTC, ETH, SOL, XRP, and ADA with 94.7% accuracy' },
                                { icon: <BarChart3 size={20} />, title: 'Real-Time Market Data', desc: 'Live prices, volume, and market cap from major exchanges' },
                                { icon: <Globe size={20} />, title: 'Portfolio Tracking', desc: 'Monitor your investments and track performance over time' },
                                { icon: <Clock size={20} />, title: 'Daily Alerts', desc: 'Get notified about significant price movements and prediction updates' },
                                { icon: <Shield size={20} />, title: 'Bank-Grade Security', desc: '256-bit encryption, 2FA, and SOC 2 compliance' },
                            ].map((benefit, i) => (
                                <div key={i} style={{ display: 'flex', gap: '16px', marginBottom: '20px' }}>
                                    <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(240, 185, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', flexShrink: 0 }}>
                                        {benefit.icon}
                                    </div>
                                    <div>
                                        <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>{benefit.title}</h3>
                                        <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>{benefit.desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Plan Preview */}
                        <div className="card card-highlight" style={{ padding: '24px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                                <div>
                                    <span style={{ fontSize: '12px', color: 'var(--accent)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Free Plan</span>
                                    <h3 style={{ fontSize: '24px', fontWeight: 700, marginTop: '4px' }}>$0 <span style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 400 }}>/month</span></h3>
                                </div>
                                <div style={{ padding: '4px 10px', background: 'rgba(14, 203, 129, 0.1)', borderRadius: '6px', fontSize: '11px', color: 'var(--green)', fontWeight: 600 }}>
                                    CURRENT
                                </div>
                            </div>
                            <ul style={{ listStyle: 'none', padding: 0 }}>
                                {[
                                    '5 AI predictions per day',
                                    'Real-time market data',
                                    'Basic portfolio tracking',
                                    'Email support',
                                ].map((feature, i) => (
                                    <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                                        <CheckCircle size={16} color="#0ecb81" />
                                        {feature}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Stats */}
                        <div style={{ display: 'flex', gap: '32px', marginTop: '32px' }}>
                            {[
                                { value: '10K+', label: 'Active Users' },
                                { value: '94.7%', label: 'Accuracy' },
                                { value: '4.9★', label: 'User Rating' },
                            ].map((stat, i) => (
                                <div key={i}>
                                    <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '24px', fontWeight: 700, color: 'var(--accent)' }}>
                                        {stat.value}
                                    </div>
                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{stat.label}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Signup;
