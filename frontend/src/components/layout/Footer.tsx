import { Link } from 'react-router-dom';

const Footer = () => {
    return (
        <footer style={{
            background: 'var(--bg-secondary)',
            borderTop: '1px solid var(--border)',
            padding: '48px 0 24px',
        }}>
            <div className="container">
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '2fr repeat(3, 1fr)',
                    gap: '48px',
                    marginBottom: '48px',
                }}>
                    {/* Brand */}
                    <div>
                        <Link to="/" style={{ display: 'inline-block', marginBottom: '16px' }}>
                            <img src="/images/logo.png" alt="CryptoVertex" style={{ height: '24px' }} />
                        </Link>
                        <p style={{ color: 'var(--text-muted)', fontSize: '13px', lineHeight: 1.7, maxWidth: '280px' }}>
                            AI-powered cryptocurrency price predictions using deep learning and 8 years of historical market data.
                        </p>
                    </div>

                    {/* Products */}
                    <div>
                        <h4 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '16px' }}>
                            Products
                        </h4>
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            {['Market Overview', 'AI Predictions', 'Price Alerts'].map(item => (
                                <li key={item} style={{ marginBottom: '10px' }}>
                                    <Link to="/market" style={{ color: 'var(--text-muted)', fontSize: '13px', textDecoration: 'none' }}>
                                        {item}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Coins */}
                    <div>
                        <h4 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '16px' }}>
                            Top Coins
                        </h4>
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            {[
                                { slug: 'btc', name: 'Bitcoin (BTC)' },
                                { slug: 'eth', name: 'Ethereum (ETH)' },
                                { slug: 'sol', name: 'Solana (SOL)' },
                                { slug: 'xrp', name: 'XRP' },
                                { slug: 'ada', name: 'Cardano (ADA)' },
                            ].map(coin => (
                                <li key={coin.slug} style={{ marginBottom: '10px' }}>
                                    <Link to={`/coin/${coin.slug}`} style={{ color: 'var(--text-muted)', fontSize: '13px', textDecoration: 'none' }}>
                                        {coin.name}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Support */}
                    <div>
                        <h4 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '16px' }}>
                            Support
                        </h4>
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            {[
                                { path: '/about', name: 'About Us' },
                                { path: '/disclaimer', name: 'Disclaimer' },
                            ].map(link => (
                                <li key={link.path} style={{ marginBottom: '10px' }}>
                                    <Link to={link.path} style={{ color: 'var(--text-muted)', fontSize: '13px', textDecoration: 'none' }}>
                                        {link.name}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                        {/* Social */}
                        <div style={{ display: 'flex', gap: '12px', marginTop: '20px' }}>
                            {[
                                { icon: 'fab fa-twitter', url: 'https://twitter.com/adnankx' },
                                { icon: 'fab fa-github', url: 'https://github.com/adnankx7' },
                                { icon: 'fab fa-telegram-plane', url: 'https://t.me/hammad-sikandar' },
                                { icon: 'fab fa-linkedin', url: 'https://www.linkedin.com/in/hammad-sikandar' },
                            ].map((social, i) => (
                                <a
                                    key={i}
                                    href={social.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    style={{
                                        width: '32px',
                                        height: '32px',
                                        borderRadius: '6px',
                                        background: 'var(--bg-card)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        color: 'var(--text-muted)',
                                        fontSize: '14px',
                                        border: '1px solid var(--border)',
                                        textDecoration: 'none',
                                    }}
                                >
                                    <i className={social.icon}></i>
                                </a>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Bottom */}
                <div style={{
                    borderTop: '1px solid var(--border)',
                    paddingTop: '20px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                }}>
                    <p style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                        © 2025 CryptoVertex. All rights reserved.
                    </p>
                    <p style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                        Predictions are for informational purposes only.
                    </p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
