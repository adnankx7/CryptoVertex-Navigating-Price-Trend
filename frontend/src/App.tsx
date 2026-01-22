import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import Market from './pages/Market';
import About from './pages/About';
import Disclaimer from './pages/Disclaimer';
import Login from './features/auth/Login';
import Signup from './features/auth/Signup';
import CoinDetail from './features/prediction/CoinDetail';
import SentimentDashboard from './features/analytics/SentimentDashboard';
import WhaleWatch from './features/analytics/WhaleWatch';
import PatternScanner from './features/analytics/PatternScanner';
import CorrelationMatrix from './features/analytics/CorrelationMatrix';
import TokenUnlocks from './features/analytics/TokenUnlocks';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/market" element={<Market />} />
        <Route path="/about" element={<About />} />
        <Route path="/disclaimer" element={<Disclaimer />} />
        <Route path="/coin/:slug" element={<CoinDetail />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/sentiment" element={<SentimentDashboard />} />
        <Route path="/whale-watch" element={<WhaleWatch />} />
        <Route path="/patterns" element={<PatternScanner />} />
        <Route path="/correlation" element={<CorrelationMatrix />} />
        <Route path="/unlocks" element={<TokenUnlocks />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
