from flask import Flask, request, jsonify, render_template, url_for
import joblib
import pandas as pd
import requests
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image
from io import BytesIO

load_dotenv()

app = Flask(__name__, static_folder='static')

# Ensure static folder exists
os.makedirs('static/images', exist_ok=True)

# Load model, training labels, and preprocessing objects
model_data = joblib.load('models/knn_model.pkl')
knn_model = model_data['model']
y_train = model_data['y_train']  # Original training labels
scaler = joblib.load('models/scaler.pkl')
ohe = joblib.load('models/ohe.pkl')

# Load clustering model
cluster_data = joblib.load('models/cluster_model.pkl')
clustering = cluster_data['model']
cluster_scaler = cluster_data['scaler']
crop_clusters = cluster_data['crop_clusters']

# Load dataset for median pH calculation
df = pd.read_csv('Crop_recommendation.csv')  # Replace with your dataset
median_ph = df['ph'].median()

# Fertilizer database (real compositions)
FERTILIZER_DB = {
    "Urea (46-0-0)": {"N": 46, "P": 0, "K": 0},
    "DAP (18-46-0)": {"N": 18, "P": 46, "K": 0},
    "MOP (0-0-60)": {"N": 0, "P": 0, "K": 60},
    "NPK 10-26-26": {"N": 10, "P": 26, "K": 26}
}

# OpenWeatherMap API (get free key: https://openweathermap.org/api)
OWM_API_KEY = os.getenv('OWM_API_KEY', 'e6ea98b7ebe98a35a2408d05e0e2e46f')

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyA3g2I-fojjpQe7PKGyAfNyuXvlY1hEgI0')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')  # Updated model version

# Add Pexels API key to your environment variables
PEXELS_API_KEY = os.getenv('OPwkrG284pQMF30G41QenRzehC4kUYHSkWkqESF8qTFcMuvq1vYBUy1x')

def get_soil_ph(lat, lon):
    """Fetch soil pH data from SoilGrids API"""
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=phh2o&depth=0-5cm&value=mean"
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Extract pH value (divide by 10 as SoilGrids returns pH * 10)
        ph_value = data['properties']['layers'][0]['depths'][0]['values']['mean'] / 10
        return {'status': 'success', 'ph': round(ph_value, 2)}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def get_weather(location):
    """Fetch weather data including annual rainfall"""
    try:
        # Get coordinates first
        geocoding_url = f'http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OWM_API_KEY}'
        geo_response = requests.get(geocoding_url)
        geo_data = geo_response.json()
        
        if not geo_data:
            return {'status': 'error', 'message': 'Location not found'}
        
        lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
        
        # Get current weather
        weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric'
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        
        # Use dataset's annual average rainfall
        annual_rainfall = df['rainfall'].mean()
        
        return {
            'temperature': weather_data['main']['temp'],
            'humidity': weather_data['main']['humidity'],
            'rainfall': round(annual_rainfall, 2),
            'status': 'success'
        }
    except Exception as e:
        print(f"Weather API Error: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def calculate_npk(fertilizer, quantity_kg, area_m2):
    """Calculate NPK values from fertilizer composition with safety checks"""
    comp = FERTILIZER_DB[fertilizer]
    area_ha = area_m2 / 10000  # Convert m² to hectares
    
    # Validate inputs
    if area_ha <= 0 or quantity_kg <= 0:
        raise ValueError("Area and quantity must be positive values")
    
    # Calculate NPK with realistic limits
    n = max(0, min(150, (comp["N"]/100) * quantity_kg / area_ha))
    p = max(0, min(150, (comp["P"]/100) * quantity_kg / area_ha))
    k = max(0, min(150, (comp["K"]/100) * quantity_kg / area_ha))
    
    return round(n, 2), round(p, 2), round(k, 2)

def get_crop_image(crop_name):
    """Fetch crop image using multiple services"""
    try:
        # Try Pexels API first
        url = "https://api.pexels.com/v1/search"
        headers = {
            "Authorization": PEXELS_API_KEY
        }
        params = {
            "query": f"{crop_name} plant agriculture field",
            "per_page": 1,
            "size": "medium",
            "orientation": "square"
        }
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get('photos'):
            return {
                'status': 'success',
                'image_url': data['photos'][0]['src']['medium']
            }
        
        # If Pexels fails, try direct Unsplash URL
        unsplash_url = f"https://source.unsplash.com/400x400/?{crop_name},agriculture"
        response = requests.head(unsplash_url)
        if response.status_code == 200:
            return {
                'status': 'success',
                'image_url': unsplash_url
            }
            
        # Final fallback to placeholder
        return {
            'status': 'success',
            'image_url': f'https://placehold.co/400x400/png?text={crop_name}'
        }
            
    except Exception as e:
        print(f"Image fetch error for {crop_name}: {str(e)}")
        return {
            'status': 'error',
            'image_url': f'https://placehold.co/400x400/png?text={crop_name}'
        }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if data.get('location') == 'manual':
            # Prepare and scale input data
            input_data = {
                'N': float(data['n']),
                'P': float(data['p']),
                'K': float(data['k']),
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'ph': float(data['ph']),
                'rainfall': float(data['rainfall'])
            }
            
            columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            input_df = pd.DataFrame([input_data], columns=columns)
            scaled_input = scaler.transform(input_df)
            
            # Get predictions with enough neighbors to find unique crops
            n_neighbors = 5  # Increased to ensure we find enough unique crops
            distances, indices = knn_model.kneighbors(scaled_input, n_neighbors=n_neighbors)
            
            # Get unique recommendations
            recommendations = []
            seen = set()
            
            for idx in indices[0]:
                crop = y_train[idx]
                if crop not in seen and len(recommendations) < 5:
                    recommendations.append(crop)
                    seen.add(crop)
                if len(recommendations) == 5:
                    break
            
            return jsonify({
                'input_data': input_data,
                'recommendations': recommendations
            })
        else:
            # Handle API-based prediction
            weather = get_weather(data['location'])
            if weather['status'] != 'success':
                return jsonify({'error': f"Weather API Error: {weather['message']}"}), 500

            # Process NPK values
            if data['has_soil_report']:
                if None in [data.get('n'), data.get('p'), data.get('k')]:
                    return jsonify({'error': 'All soil report fields are required'}), 400
                n, p, k = data['n'], data['p'], data['k']
                ph = data['ph'] if data.get('ph') else weather.get('ph', median_ph)
            else:
                if None in [data.get('fertilizer'), data.get('quantity_kg'), data.get('area_m2')]:
                    return jsonify({'error': 'All fertilizer fields are required'}), 400
                try:
                    n, p, k = calculate_npk(data['fertilizer'], data['quantity_kg'], data['area_m2'])
                except ValueError as e:
                    return jsonify({'error': str(e)}), 400
                ph = weather.get('ph', median_ph)

            # Prepare input data
            input_data = {
                'N': n,
                'P': p,
                'K': k,
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'ph': ph,
                'rainfall': weather['rainfall']
            }
            
            # Scale and predict
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df)
            n_neighbors = 5
            distances, indices = knn_model.kneighbors(scaled_input, n_neighbors=n_neighbors)

            # Get unique recommendations
            recommendations = []
            seen = set()
            
            for idx in indices[0]:
                if idx < len(y_train):
                    crop = y_train[idx]
                    if crop not in seen and len(recommendations) < 5:
                        recommendations.append(crop)
                        seen.add(crop)
                    if len(recommendations) == 5:
                        break

            return jsonify({
                'input_data': input_data,
                'recommendations': recommendations,
                'weather_status': 'Fetched from OpenWeatherMap API'
            })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/current-weather/<location>')
def current_weather(location):
    try:
        lat, lon = location.split(',')
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            # Calculate precipitation chance based on clouds and humidity
            precipitation = min(100, (data['clouds']['all'] + data['main']['humidity']) / 2)
            
            return jsonify({
                'status': 'success',
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'humidity': data['main']['humidity'],
                'precipitation': precipitation,
                'rainfall': data.get('rain', {}).get('1h', 0)
            })
        
        return jsonify({'error': 'Failed to fetch weather data'})

    except Exception as e:
        print("Weather API Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/manual-input')
def manual_input():
    """Render manual input form"""
    return render_template('manual_input.html')

@app.route('/recommendations')
def show_recommendations():
    """Route to display recommendations"""
    return render_template('recommendations.html')

@app.route('/companion-crops', methods=['POST'])
def get_companion_crops():
    try:
        data = request.json
        crop_name = data.get('crop')
        
        # Find the cluster for the input crop
        crop_info = next((item for item in crop_clusters if item['label'] == crop_name), None)
        
        if not crop_info:
            return jsonify({'error': 'Crop not found'}), 404
            
        cluster_id = crop_info['cluster']
        
        # Get unique companion crops using set
        companion_crops = list({
            item['label'] 
            for item in crop_clusters 
            if item['cluster'] == cluster_id and item['label'] != crop_name
        })
        
        # Sort alphabetically for consistent output
        companion_crops.sort()
        
        # Take only top 5 if more exist
        companion_crops = companion_crops[:5]
        
        # Only include growing_conditions if no companion crops found
        response_data = {
            'input_crop': crop_name,
            'companion_crops': companion_crops
        }
        
        if not companion_crops:
            response_data['growing_conditions'] = 'No companion crops found with similar growing conditions'
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/companion-search')
def companion_search():
    """Render companion crops search page"""
    return render_template('companion_search.html')

@app.route('/location-input')
def location_input():
    return render_template('location_input.html')

@app.route('/location')
def location_based():
    """Route for location based recommendations"""
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)