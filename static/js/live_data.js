// Cached currency formatter
const usdFormatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
});

function formatPrice(price) {
    return usdFormatter.format(price);
}

const coinDataMap = {
    'BTC/USDT': { name: 'bitcoin', symbol: 'btc' },
    'ETH/USDT': { name: 'ethereum', symbol: 'eth' },
    'ADA/USDT': { name: 'cardano', symbol: 'ada' },
    'SOL/USDT': { name: 'solana', symbol: 'sol' },
    'XRP/USDT': { name: 'xrp', symbol: 'xrp' }
};

// Map URL path to coin symbol
function getCurrentCoinSymbol() {
    const path = window.location.pathname.toLowerCase();
    const coinMap = {
        '/btc': 'BTC/USDT',
        '/eth': 'ETH/USDT',
        '/sol': 'SOL/USDT',
        '/xrp': 'XRP/USDT',
        '/ada': 'ADA/USDT'
    };
    return coinMap[path] || null;
}

function updatePriceElements(symbol, data) {
    document.querySelectorAll(`[data-symbol="${symbol}"][data-field="price"]`).forEach(element => {
        element.textContent = data.price ? formatPrice(data.price) : 'N/A';
    });

    document.querySelectorAll(`[data-symbol="${symbol}"][data-field="change"]`).forEach(element => {
        if (data.change !== null) {
            element.textContent = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)}%`;
            element.className = data.change >= 0 ? 'up' : 'down';
        } else {
            element.textContent = 'N/A';
        }
    });
}

function updateTopSection(sectionId, items) {
    const container = document.getElementById(sectionId);
    container.innerHTML = items.map(([symbol, data]) => {
        const coinInfo = coinDataMap[symbol];
        return `
            <li>
                <img src="/static/images/${coinInfo.name}-${coinInfo.symbol}-logo.png" 
                     class="coin-logo" 
                     alt="${coinInfo.name} logo"
                     onerror="this.src='/static/images/default-coin.png'">
                <span>${symbol.split('/')[0]}</span>
                <span data-symbol="${symbol}" data-field="price">
                    ${data.price ? formatPrice(data.price) : 'N/A'}
                </span>
                <span class="${data.change >= 0 ? 'up' : 'down'}" 
                      data-symbol="${symbol}" 
                      data-field="change">
                    ${data.change !== null ? `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)}%` : 'N/A'}
                </span>
            </li>
        `;
    }).join('');
}

async function fetchLiveData() {
    try {
        const response = await fetch('/live_data');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch live data:', error);
        showErrorMessage();
        return null;
    }
}

function showErrorMessage() {
    document.querySelectorAll('[data-field]').forEach(el => {
        if (el.textContent === 'Loading...') {
            el.textContent = 'Data unavailable';
            el.style.color = '#FF4F5A';
        }
    });
}

// Mock historical data for chart (until real API is available)
function generateMockHistory(currentPrice, points = 24) {
    const history = [];
    let price = currentPrice;
    for (let i = 0; i < points; i++) {
        price *= (1 + (Math.random() * 0.02 - 0.01)); // ±1% change
        history.push(parseFloat(price.toFixed(2)));
    }
    return history;
}

// Unified update function
async function updateAllData() {
    const data = await fetchLiveData();
    if (!data) return;

    const currentCoin = getCurrentCoinSymbol();

    if (currentCoin) {
        const coinData = data.live_data[currentCoin];
        updatePriceElements(currentCoin, coinData);

        const mockHistory = generateMockHistory(coinData.price);
        if (window.updateChart) updateChart(mockHistory);
    } else {
        Object.entries(data.live_data).forEach(([symbol, coinData]) => {
            updatePriceElements(symbol, coinData);
        });
        updateTopSection('top-gainers-list', data.top_gainers);
        updateTopSection('top-losers-list', data.top_losers);
    }
}

// DOM ready
document.addEventListener('DOMContentLoaded', () => {
    updateAllData();

    // Update every second
    setInterval(() => {
        if (!document.hidden) updateAllData();
    }, 1000);

    // Show error if data doesn't load in 10s
    setTimeout(() => {
        if ([...document.querySelectorAll('[data-field]')].some(el => el.textContent === 'Loading...')) {
            showErrorMessage();
        }
    }, 10000);
});