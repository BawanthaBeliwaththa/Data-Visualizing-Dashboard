import sys
import os
from flask import Flask, render_template, jsonify, request, make_response, send_file
from flask_cors import CORS
import logging
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.data_loader import DataLoader
from models.preprocessing import DataPreprocessor
from models.visualizations import Visualizer
from models.analysis import Analyzer

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Add custom template filters
@app.template_filter('format_number')
def format_number_filter(value):
    """Format number with commas as thousands separators"""
    try:
        if value is None:
            return 'N/A'
        if isinstance(value, (int, float)):
            return f"{value:,.0f}".replace(',', ',')
        return str(value)
    except (ValueError, TypeError):
        return str(value) if value is not None else 'N/A'

# Global variables to store data
data_loader = DataLoader()
preprocessor = None
visualizer = None
analyzer = None
processed_data = None  # Store processed data for reuse

def initialize_data(force=False):
    """Initialize data and models"""
    global preprocessor, visualizer, analyzer, processed_data
    
    try:
        # Check if data is already initialized
        if processed_data is not None and not force:
            logger.info("Using cached processed data")
            return True
            
        # Load raw data
        raw_data = data_loader.load_data(force_reload=force)
        
        # Preprocess data
        preprocessor = DataPreprocessor(raw_data)
        processed_data = preprocessor.preprocess()
        
        # Initialize visualizer and analyzer
        visualizer = Visualizer(processed_data)
        analyzer = Analyzer(processed_data)
        
        logger.info("Data initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        return False

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    if not initialize_data():
        return render_template('error.html', message="Failed to load data")
    
    stats = analyzer.get_summary_statistics()
    trends = analyzer.get_yearly_trends()
    top_countries = analyzer.get_top_countries()
    
    return render_template('dashboard.html', 
                         stats=stats,
                         trends=trends,
                         top_countries=top_countries)

@app.route('/visualizations')
def visualizations():
    """Visualizations page"""
    if not initialize_data():
        return render_template('error.html', message="Failed to load data")
    
    return render_template('visualizations/index.html')

@app.route('/api/visualization/<chart_type>')
def get_visualization(chart_type):
    """API endpoint to get visualization JSON"""
    if not initialize_data():
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        if chart_type == 'line':
            chart_json = visualizer.create_line_chart()
        elif chart_type == 'bar':
            top_n = request.args.get('top_n', 10, type=int)
            chart_json = visualizer.create_bar_chart(top_n)
        elif chart_type == 'pie':
            chart_json = visualizer.create_pie_chart()
        elif chart_type == 'correlation':
            chart_json = visualizer.create_correlation_matrix()
        elif chart_type == 'scatter':
            chart_json = visualizer.create_scatter_plot()
        elif chart_type == 'boxplot':
            chart_json = visualizer.create_boxplot()
        elif chart_type == 'region_boxplot':
            chart_json = visualizer.create_region_boxplot()
        else:
            return jsonify({'error': 'Invalid chart type'}), 400
        
        if chart_json is None:
            return jsonify({'error': 'Chart data not available'}), 404
            
        return jsonify({'success': True, 'chart': chart_json})
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_type>')
def get_analysis(analysis_type):
    """API endpoint to get analysis data"""
    if not initialize_data():
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        if analysis_type == 'summary':
            data = analyzer.get_summary_statistics()
        elif analysis_type == 'trends':
            data = analyzer.get_yearly_trends()
        elif analysis_type == 'top_countries':
            n = request.args.get('n', 10, type=int)
            data = analyzer.get_top_countries(n=n)
        elif analysis_type == 'regional':
            data = analyzer.get_regional_summary()
        elif analysis_type == 'mdr_trend':
            data = analyzer.get_mdr_trend()
        elif analysis_type == 'correlation':
            data = analyzer.get_correlation_analysis()
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"Error getting analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/preview')
def data_preview():
    """Get data preview"""
    if not initialize_data():
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Get preview from processed data
        preview_data = []
        for idx, row in processed_data.head(20).iterrows():
            preview_data.append({
                'country': row.get('country', 'N/A'),
                'year': int(row.get('year', 0)) if pd.notna(row.get('year', 0)) else 'N/A',
                'pulm_labconf_new': float(row.get('pulm_labconf_new', 0)) if pd.notna(row.get('pulm_labconf_new', 0)) else None,
                'mdr_new': float(row.get('mdr_new', 0)) if pd.notna(row.get('mdr_new', 0)) else None,
                'xdr': float(row.get('xdr', 0)) if pd.notna(row.get('xdr', 0)) else None
            })
        
        info = {
            'shape': processed_data.shape,
            'unique_countries': processed_data['country'].nunique() if 'country' in processed_data.columns else 0,
            'columns': list(processed_data.columns),
            'year_range': [int(processed_data['year'].min()), int(processed_data['year'].max())] if 'year' in processed_data.columns else [0, 0]
        }
        
        return jsonify({'success': True, 'preview': preview_data, 'info': info})
    except Exception as e:
        logger.error(f"Error getting data preview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/refresh-data')
def refresh_data():
    """Force refresh data from source"""
    global preprocessor, visualizer, analyzer, processed_data
    
    try:
        # Force reload data
        raw_data = data_loader.load_data(force_reload=True)
        preprocessor = DataPreprocessor(raw_data)
        processed_data = preprocessor.preprocess()
        visualizer = Visualizer(processed_data)
        analyzer = Analyzer(processed_data)
        return jsonify({'success': True, 'message': 'Data refreshed successfully'})
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/report')
def export_report():
    """Export complete HTML report"""
    if not initialize_data():
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Generate HTML report
        html_content = visualizer.generate_report()
        
        # Create response with proper headers
        response = make_response(html_content)
        response.headers['Content-Type'] = 'text/html'
        response.headers['Content-Disposition'] = f'attachment; filename=tb_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        return response
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/csv')
def export_csv():
    """Export data as CSV"""
    if not initialize_data():
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Convert DataFrame to CSV
        csv_data = processed_data.to_csv(index=False)
        
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=tb_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="Internal server error"), 500

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)