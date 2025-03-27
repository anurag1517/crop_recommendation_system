from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import requests
import numpy as np

app = Flask(__name__)

# Load model, training labels, and preprocessing objects
model_data = joblib.load('models/knn_model.pkl')
knn_model = model_data['model']
y_train = model_data['y_train']  # Original training labels
scaler = joblib.load('models/scaler.pkl')
ohe = joblib.load('models/ohe.pkl')

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
OWM_API_KEY = "e6ea98b7ebe98a35a2408d05e0e2e46f"

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

@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/current-weather/<location>', methods=['GET'])
def current_weather(location):
    """Get current weather for location"""
    try:
        weather = get_weather(location)
        if weather['status'] == 'success':
            return jsonify({
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'status': 'success'
            })
        return jsonify({'error': 'Failed to fetch weather data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/manual-input')
def manual_input():
    """Render manual input form"""
    return render_template('manual_input.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)