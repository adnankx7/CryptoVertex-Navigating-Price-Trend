// Debounce function to limit API calls
function debounce(func, delay) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
}

const searchInput = document.getElementById('coin-search-input');
const suggestionsList = document.getElementById('search-suggestions');

function clearSuggestions() {
    suggestionsList.innerHTML = '';
    suggestionsList.style.display = 'none';
}

function createSuggestionItem(coin) {
    const li = document.createElement('li');
    li.textContent = `${coin.full_name} (${coin.symbol.split('/')[0]})`;
    li.style.padding = '8px';
    li.style.cursor = 'pointer';
    li.style.backgroundColor = '#1a1f2c';
    li.addEventListener('click', () => {
        window.location.href = coin.url;
    });
    return li;
}

async function fetchSuggestions(query) {
    if (!query) {
        clearSuggestions();
        return;
    }
    try {
        const response = await fetch(`/search_coins?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Network response was not ok');
        const results = await response.json();
        suggestionsList.innerHTML = '';
        if (results.length === 0) {
            clearSuggestions();
            return;
        }
        results.forEach(coin => {
            const item = createSuggestionItem(coin);
            suggestionsList.appendChild(item);
        });
        suggestionsList.style.display = 'block';
    } catch (error) {
        console.error('Error fetching search suggestions:', error);
        clearSuggestions();
    }
}

const debouncedFetch = debounce(fetchSuggestions, 300);

searchInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    debouncedFetch(query);
});

// Hide suggestions when clicking outside
document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !suggestionsList.contains(e.target)) {
        clearSuggestions();
    }
});
