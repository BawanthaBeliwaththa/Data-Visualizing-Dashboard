// Main JavaScript file for TB Data Analysis Platform

// Global variables
let charts = {};

// Initialize on document load
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    initializeCharts();
});

// Initialize tooltips
function initializeTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

// Show tooltip
function showTooltip(e) {
    const tooltip = document.createElement('div');
    tooltip.className = 'absolute bg-gray-800 text-white px-2 py-1 rounded text-sm z-50';
    tooltip.textContent = e.target.dataset.tooltip;
    tooltip.style.left = e.pageX + 'px';
    tooltip.style.top = (e.pageY - 30) + 'px';
    tooltip.id = 'tooltip';
    document.body.appendChild(tooltip);
}

// Hide tooltip
function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// Initialize charts that might be on the page
function initializeCharts() {
    // Check if we're on the visualizations page
    if (document.getElementById('line-chart')) {
        loadChart('line', 'line-chart');
    }
    if (document.getElementById('bar-chart')) {
        loadChart('bar', 'bar-chart');
    }
    if (document.getElementById('pie-chart')) {
        loadChart('pie', 'pie-chart');
    }
    if (document.getElementById('correlation-chart')) {
        loadChart('correlation', 'correlation-chart');
    }
    if (document.getElementById('scatter-chart')) {
        loadChart('scatter', 'scatter-chart');
    }
    if (document.getElementById('region-boxplot')) {
        loadChart('region_boxplot', 'region-boxplot');
    }
}

// Load a specific chart
function loadChart(type, elementId) {
    showLoading(elementId);
    
    fetch(`/api/visualization/${type}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const chartData = JSON.parse(data.chart);
                charts[elementId] = Plotly.newPlot(elementId, chartData.data, chartData.layout);
            } else {
                showError(elementId, 'Failed to load chart');
            }
        })
        .catch(error => {
            console.error('Error loading chart:', error);
            showError(elementId, 'Error loading chart');
        });
}

// Show loading spinner
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="flex justify-center items-center h-full">
                <div class="loader"></div>
            </div>
        `;
    }
}

// Show error message
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="flex justify-center items-center h-full">
                <div class="text-red-500 text-center">
                    <i class="fas fa-exclamation-circle text-4xl mb-2"></i>
                    <p>${message}</p>
                </div>
            </div>
        `;
    }
}

// Format number with commas
function formatNumber(num) {
    if (num === null || num === undefined) return 'N/A';
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Export data to CSV
function exportToCSV(data, filename) {
    const csv = convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Convert JSON to CSV
function convertToCSV(objArray) {
    const array = typeof objArray !== 'object' ? JSON.parse(objArray) : objArray;
    let str = '';
    
    for (let i = 0; i < array.length; i++) {
        let line = '';
        for (let index in array[i]) {
            if (line !== '') line += ',';
            line += array[i][index];
        }
        str += line + '\r\n';
    }
    return str;
}

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for performance
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Handle window resize with debounce
window.addEventListener('resize', debounce(function() {
    // Redraw charts on resize
    Object.keys(charts).forEach(elementId => {
        Plotly.Plots.resize(elementId);
    });
}, 250));

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl + R to refresh data
    if (e.ctrlKey && e.key === 'r') {
        e.preventDefault();
        refreshData();
    }
    
    // Esc to close fullscreen modal
    if (e.key === 'Escape') {
        const modal = document.getElementById('fullscreen-modal');
        if (modal && !modal.classList.contains('hidden')) {
            closeFullscreen();
        }
    }
});

// Handle offline/online status
window.addEventListener('online', function() {
    showNotification('Connection restored', 'success');
});

window.addEventListener('offline', function() {
    showNotification('You are offline. Some features may be unavailable.', 'warning');
});

// Service worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js').then(function(registration) {
            console.log('ServiceWorker registered');
        }, function(err) {
            console.log('ServiceWorker registration failed: ', err);
        });
    });
}

// Export functions for use in HTML
window.formatNumber = formatNumber;
window.exportToCSV = exportToCSV;
window.showNotification = showNotification;
