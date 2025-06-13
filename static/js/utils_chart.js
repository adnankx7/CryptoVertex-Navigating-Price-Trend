let priceChart;
const MAX_RETRIES = 3;
let retryCount = 0;

async function loadHistoricalData(coinSymbol) {
    try {
        const encodedSymbol = encodeURIComponent(coinSymbol);
        const response = await fetch(`/historical_data/${encodedSymbol}`);

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Data file not found on server');
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('Empty dataset received');
        }

        return data.map(entry => ({
            x: entry.Date,
            y: entry.Close
        }));

    } catch (error) {
        console.error('Data loading error:', error);
        throw error;
    }
}

async function initializeHistoricalChart(coinSymbol, chartColor) {
    try {
        document.getElementById('chartLoading').style.display = 'block';
        document.getElementById('chartError').style.display = 'none';

        const chartData = await loadHistoricalData(coinSymbol);
        const ctx = document.getElementById('priceChart').getContext('2d');

        if (priceChart) {
            priceChart.destroy();
        }

        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Daily Closing Price',
                    data: chartData,
                    borderColor: chartColor,
                    backgroundColor: `${chartColor}1A`, // Add alpha for background color
                    tension: 0.1,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            tooltipFormat: 'MMM dd, yyyy',
                            displayFormats: {
                                day: 'MMM dd'
                            }
                        },
                        grid: { color: '#2A3038' },
                        ticks: { color: '#E0E0E0' }
                    },
                    y: {
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString('en-US', {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2
                                });
                            },
                            color: '#E0E0E0'
                        },
                        grid: { color: '#2A3038' }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#E0E0E0',
                            font: { size: 14 }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1a1f2c',
                        titleColor: chartColor,
                        bodyColor: '#E0E0E0',
                        borderColor: '#2A3038',
                        borderWidth: 1,
                        bodySpacing: 5,
                        callbacks: {
                            label: (context) => {
                                const value = context.parsed.y;
                                return `Price: $${value.toFixed(2)} (${context.dataset.label})`;
                            }
                        }
                    }
                }
            }
        });

    } catch (error) {
        if (retryCount < MAX_RETRIES) {
            retryCount++;
            console.log(`Retrying... Attempt ${retryCount}/${MAX_RETRIES}`);
            setTimeout(() => initializeHistoricalChart(coinSymbol, chartColor), 2000);
        } else {
            document.getElementById('chartError').style.display = 'block';
            console.error('Final chart initialization error:', error);
        }
    } finally {
        document.getElementById('chartLoading').style.display = 'none';
    }
}