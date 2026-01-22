# CryptoVertex: Professional Crypto Intelligence Platform

CryptoVertex is an advanced cryptocurrency analytics and forecasting platform powered by AI. It bridges the gap between institutional-grade tools and retail traders, offering real-time market data, sentiment analysis, and predictive models in a unified, professional interface.

## 🚀 Features

### 🧠 AI & Analytics
- **Social Sentiment Engine**: Real-time "Fear & Greed" analysis derived from social signals (Twitter, Reddit).
- **Pattern Scanner**: AI-powered detection of classical chart patterns (Bull Flags, Head & Shoulders) with confidence scores.
- **Correlation Matrix**: Interactive heatmap showing asset correlations to help diversify portfolios.
- **On-Chain Whale Watcher**: Live tracking of large wallet movements and exchange flows.
- **Token Unlock Calendar**: Vesting schedule tracker to anticipate supply shocks.

### 📈 Market Data
- **Real-Time Prices**: Live dashboards for top gainers, losers, and market trends.
- **Coin Details**: Deep dives into individual assets with historical data and price predictions.
- **Prediction Models**: ML-based price forecasting for major cryptocurrencies.

## 🏗️ Tech Stack

### Frontend
- **Framework**: React 18 + Vite
- **Language**: TypeScript
- **Styling**: Modern CSS Variables, Glassmorphism Design
- **Icons**: Lucide React
- **Charts**: Recharts

### Backend
- **Framework**: FastAPI (Python)
- **ML/AI**: Scikit-Learn (Simulated for Demo), Pandas
- **Authentication**: JWT (JSON Web Tokens)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Backend Setup
```bash
cd backend
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```
*Server runs at `http://localhost:8000`*

### 2. Frontend Setup
```bash
cd frontend
# Install dependencies
npm install

# Start development server
npm run dev
```
*App runs at `http://localhost:5173`*

## 🐳 Docker Support
You can also run the entire stack using Docker Compose:
```bash
docker-compose up --build
```

## 📄 License
This project is licensed under the MIT License.
