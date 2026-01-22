import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { authApi } from '../../services/api';
import Navbar from '../../components/layout/Navbar';

import { Mail, Lock, Shield, Zap, BarChart3, Star, ArrowRight } from 'lucide-react';

const Login = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        try {
            const res = await authApi.login(formData);
            localStorage.setItem('token', res.data.access_token);
            navigate('/');
        } catch {
            setError('Invalid credentials. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="bg-grid" />
            <Navbar />

            <main style={{ flex: 1, display: 'flex', position: 'relative', zIndex: 1 }}>
                {/* Left Side - Features */}
                <div style={{
                    flex: 1,
                    background: 'var(--bg-secondary)',
                    borderRight: '1px solid var(--border)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    padding: '60px',
                }}>
                    <div style={{ maxWidth: '480px' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(240, 185, 11, 0.1)', padding: '6px 14px', borderRadius: '50px', marginBottom: '24px' }}>
                            <Zap size={14} color="#f0b90b" />
                            <span style={{ fontSize: '12px', color: '#f0b90b', fontWeight: 600 }}>AI-Powered Trading</span>
                        </div>

                        <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '36px', fontWeight: 700, marginBottom: '16px' }}>
                            Welcome Back to <span style={{ color: 'var(--accent)' }}>CryptoVertex</span>
                        </h1>

                        <p style={{ color: 'var(--text-secondary)', fontSize: '16px', lineHeight: 1.7, marginBottom: '40px' }}>
                            Access your personalized dashboard, view AI predictions, and make smarter trading decisions with institutional-grade analytics.
                        </p>

                        {/* Features List */}
                        <div style={{ marginBottom: '40px' }}>
                            {[
                                { icon: <BarChart3 size={20} />, title: 'AI Price Predictions', desc: '94.7% accuracy on daily forecasts' },
                                { icon: <Shield size={20} />, title: 'Secure Platform', desc: '256-bit encryption & 2FA' },
                                { icon: <Zap size={20} />, title: 'Real-Time Data', desc: 'Live prices from major exchanges' },
                            ].map((feature, i) => (
                                <div key={i} style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
                                    <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(240, 185, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', flexShrink: 0 }}>
                                        {feature.icon}
                                    </div>
                                    <div>
                                        <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>{feature.title}</h3>
                                        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{feature.desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Testimonial */}
                        <div className="card" style={{ padding: '24px' }}>
                            <div style={{ display: 'flex', gap: '4px', marginBottom: '12px' }}>
                                {[1, 2, 3, 4, 5].map(i => <Star key={i} size={14} color="#f0b90b" fill="#f0b90b" />)}
                            </div>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.7, marginBottom: '16px', fontStyle: 'italic' }}>
                                "CryptoVertex has transformed how I trade. The AI predictions are incredibly accurate and have helped me make much better decisions."
                            </p>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--bg-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, color: 'var(--accent)' }}>
                                    A
                                </div>
                                <div>
                                    <div style={{ fontSize: '14px', fontWeight: 600 }}>Alex Chen</div>
                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Day Trader • Member since 2023</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Side - Form */}
                <div style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '60px',
                }}>
                    <div style={{ width: '100%', maxWidth: '400px' }}>
                        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
                            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '28px', fontWeight: 700, marginBottom: '8px' }}>
                                Sign In
                            </h2>
                            <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                                Enter your credentials to access your account
                            </p>
                        </div>

                        <div className="card" style={{ padding: '32px' }}>
                            <form onSubmit={handleSubmit}>
                                {error && (
                                    <div style={{ background: 'rgba(246, 70, 93, 0.1)', border: '1px solid rgba(246, 70, 93, 0.3)', borderRadius: '10px', padding: '12px 16px', marginBottom: '20px', color: 'var(--red)', fontSize: '13px' }}>
                                        {error}
                                    </div>
                                )}

                                <div style={{ marginBottom: '20px' }}>
                                    <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>
                                        Email Address
                                    </label>
                                    <div style={{ position: 'relative' }}>
                                        <Mail size={18} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                                        <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="input" placeholder="you@example.com" style={{ paddingLeft: '44px' }} required />
                                    </div>
                                </div>

                                <div style={{ marginBottom: '24px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                        <label style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>Password</label>
                                        <a href="#" style={{ fontSize: '12px' }}>Forgot password?</a>
                                    </div>
                                    <div style={{ position: 'relative' }}>
                                        <Lock size={18} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                                        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="input" placeholder="••••••••" style={{ paddingLeft: '44px' }} required />
                                    </div>
                                </div>

                                <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: '100%', justifyContent: 'center', padding: '14px', fontSize: '15px' }}>
                                    {loading ? 'Signing in...' : <>Sign In <ArrowRight size={16} /></>}
                                </button>

                                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', margin: '24px 0' }}>
                                    <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>or continue with</span>
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
                            Don't have an account?{' '}
                            <Link to="/signup" style={{ fontWeight: 600 }}>Create one free</Link>
                        </p>

                        {/* Trust Badges */}
                        <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', marginTop: '32px' }}>
                            {[
                                { icon: <Shield size={16} />, text: 'SSL Secured' },
                                { icon: <Lock size={16} />, text: '2FA Available' },
                            ].map((badge, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '12px' }}>
                                    {badge.icon}
                                    {badge.text}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Login;
