import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, TrendingUp, BookOpen, Activity, ChevronDown, Zap, Waves, Crosshair, Grid, Lock } from 'lucide-react';
import { useState } from 'react';

const Navbar = () => {
    const location = useLocation();
    const isActive = (path: string) => location.pathname === path;
    const [analyticsOpen, setAnalyticsOpen] = useState(false);

    return (
        <nav style={{ borderBottom: '1px solid var(--border)', background: 'rgba(11, 14, 17, 0.8)', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 100 }}>
            <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '70px' }}>
                <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '10px', textDecoration: 'none' }}>
                    <div style={{ width: '36px', height: '36px', background: 'linear-gradient(135deg, #f0b90b 0%, #fcd535 100%)', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <TrendingUp size={22} color="#000" />
                    </div>
                    <span style={{ fontSize: '20px', fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '-0.5px' }}>
                        Crypto<span style={{ color: 'var(--accent)' }}>Vertex</span>
                    </span>
                </Link>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Link to="/" className={`nav-link ${isActive('/') ? 'active' : ''}`}>
                        <LayoutDashboard size={18} /> Dashboard
                    </Link>
                    <Link to="/market" className={`nav-link ${isActive('/market') ? 'active' : ''}`}>
                        <Activity size={18} /> Markets
                    </Link>

                    <div className="nav-item-dropdown"
                        style={{ position: 'relative' }}
                        onMouseEnter={() => setAnalyticsOpen(true)}
                        onMouseLeave={() => setAnalyticsOpen(false)}>
                        <div className={`nav-link ${isActive('/sentiment') || isActive('/whale-watch') ? 'active' : ''}`} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <Zap size={18} /> Analytics <ChevronDown size={14} />
                        </div>

                        {analyticsOpen && (
                            <div style={{
                                position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
                                background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                                borderRadius: '12px', padding: '8px', minWidth: '220px',
                                boxShadow: '0 10px 40px -10px rgba(0,0,0,0.5)',
                                display: 'flex', flexDirection: 'column', gap: '4px'
                            }}>
                                <Link to="/sentiment" className="dropdown-item" style={{ padding: '10px 12px', display: 'flex', alignItems: 'center', gap: '12px', borderRadius: '8px', color: 'var(--text-primary)', textDecoration: 'none', transition: 'background 0.2s' }} onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-tertiary)'} onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                                    <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(240, 185, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)' }}>
                                        <Zap size={18} />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '14px', fontWeight: 600 }}>Sentiment</div>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Social emotion analysis</div>
                                    </div>
                                </Link>
                                <Link to="/whale-watch" className="dropdown-item" style={{ padding: '10px 12px', display: 'flex', alignItems: 'center', gap: '12px', borderRadius: '8px', color: 'var(--text-primary)', textDecoration: 'none', transition: 'background 0.2s' }} onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-tertiary)'} onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                                    <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(14, 203, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0ecb81' }}>
                                        <Waves size={18} />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '14px', fontWeight: 600 }}>Whale Watch</div>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>On-chain flow tracker</div>
                                    </div>
                                </Link>
                                <Link to="/patterns" className="dropdown-item" style={{ padding: '10px 12px', display: 'flex', alignItems: 'center', gap: '12px', borderRadius: '8px', color: 'var(--text-primary)', textDecoration: 'none', transition: 'background 0.2s' }} onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-tertiary)'} onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                                    <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(56, 189, 248, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#38bdf8' }}>
                                        <Crosshair size={18} />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '14px', fontWeight: 600 }}>Pattern Scanner</div>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Automated chart analysis</div>
                                    </div>
                                </Link>
                                <Link to="/correlation" className="dropdown-item" style={{ padding: '10px 12px', display: 'flex', alignItems: 'center', gap: '12px', borderRadius: '8px', color: 'var(--text-primary)', textDecoration: 'none', transition: 'background 0.2s' }} onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-tertiary)'} onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                                    <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(236, 72, 153, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ec4899' }}>
                                        <Grid size={18} />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '14px', fontWeight: 600 }}>Correlations</div>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Asset heatmaps</div>
                                    </div>
                                </Link>
                                <Link to="/unlocks" className="dropdown-item" style={{ padding: '10px 12px', display: 'flex', alignItems: 'center', gap: '12px', borderRadius: '8px', color: 'var(--text-primary)', textDecoration: 'none', transition: 'background 0.2s' }} onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-tertiary)'} onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}>
                                    <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(168, 85, 247, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a855f7' }}>
                                        <Lock size={18} />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '14px', fontWeight: 600 }}>Token Unlocks</div>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Vesting schedules</div>
                                    </div>
                                </Link>
                            </div>
                        )}
                    </div>

                    <Link to="/about" className={`nav-link ${isActive('/about') ? 'active' : ''}`}>
                        <BookOpen size={18} /> About
                    </Link>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <Link to="/login" className="btn btn-secondary" style={{ padding: '8px 20px', fontSize: '14px' }}>Log In</Link>
                    <Link to="/signup" className="btn btn-primary" style={{ padding: '8px 20px', fontSize: '14px' }}>Sign Up</Link>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
