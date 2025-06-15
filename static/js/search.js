document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('coin-search-input');
    const suggestionsList = document.getElementById('search-suggestions');
    let allCoins = []; // This will cache the coin data to avoid repeated API calls.

    // Fetch the list of all coins once the page is loaded and cache it.
    fetch('/api/search_coins')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            allCoins = data;
        })
        .catch(error => {
            console.error('Error fetching coin data:', error);
            // Optionally, display an error message to the user in the UI
        });

    // Add an event listener to the search input to handle user typing.
    searchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase().trim();
        
        // Clear previous suggestions
        suggestionsList.innerHTML = '';

        if (query.length === 0) {
            suggestionsList.style.display = 'none';
            return;
        }

        // Filter the cached list of coins based on the user's query.
        const filteredCoins = allCoins.filter(coin =>
            coin.name.toLowerCase().includes(query) ||
            coin.ticker.toLowerCase().includes(query)
        );

        // If there are matches, build and display the suggestions list.
        if (filteredCoins.length > 0) {
            filteredCoins.forEach(coin => {
                const listItem = document.createElement('li');
                const link = document.createElement('a');
                
                // The link's URL is created from the coin's 'slug' (e.g., /btc)
                link.href = `/${coin.slug}`;
                
                // The link's text shows the full name and ticker (e.g., "Bitcoin (BTC)")
                link.textContent = `${coin.name} (${coin.ticker})`;
                
                listItem.appendChild(link);
                suggestionsList.appendChild(listItem);
            });
            suggestionsList.style.display = 'block'; // Make the full list visible
        } else {
            suggestionsList.style.display = 'none'; // Hide if no matches are found
        }
    });

    // Add a global click listener to hide the suggestions when clicking elsewhere.
    document.addEventListener('click', function(event) {
        // Check if the click was outside of the search input and the suggestions list.
        if (!searchInput.contains(event.target) && !suggestionsList.contains(event.target)) {
            suggestionsList.style.display = 'none';
        }
    });
});