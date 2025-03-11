import os
from flask import Flask, render_template, request, jsonify, url_for
import pickle
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime

def format_number(value):
    return "{:,.2f}".format(value)

def load_latest_prices():
    try:
        prices_df = pd.read_pickle('data/latest_market_prices.pkl')
        return prices_df.to_dict(orient='index')
    except:
        return None

app = Flask(__name__)
app.jinja_env.filters['format_number'] = format_number

# Load model and data
model_data = pickle.load(open('data/crop_prediction_model.pkl', 'rb'))
model = model_data['model']
label_encoder_soil = model_data['label_encoder_soil']
label_encoder_period = model_data['label_encoder_period']
label_encoder_crop = model_data['label_encoder_crop']
scaler = model_data['scaler']

# Define feature names
feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 
                 'Humidity', 'pH', 'Soil Type', 'Period of Month', 
                 'NPK_Ratio', 'Temp_Humidity_Index', 'Soil_Moisture_Index', 
                 'NPK_TH_Index']

def update_crop_prices():
    """Update CROP_DATA with latest prices from pkl file"""
    market_prices = load_latest_prices()
    if market_prices:
        for crop, price_info in market_prices.items():
            if crop in CROP_DATA:
                # Convert price based on unit
                if 'CWT' in price_info['unit']:
                    price_per_kg = price_info['price_inr'] / 45.36
                elif 'BU' in price_info['unit']:
                    price_per_kg = price_info['price_inr'] / 27.2155
                elif 'LB' in price_info['unit']:
                    price_per_kg = price_info['price_inr'] * 2.20462
                elif 'TON' in price_info['unit']:
                    price_per_kg = price_info['price_inr'] / 1000
                else:
                    price_per_kg = price_info['price_inr']
                
                CROP_DATA[crop]['price_per_kg'] = price_per_kg
                CROP_DATA[crop]['last_updated'] = price_info['timestamp']

# Initial CROP_DATA dictionary with yield data
CROP_DATA = {
    'Rice': {'yield_per_acre': 30},
    'Wheat': {'yield_per_acre': 25},
    'Corn': {'yield_per_acre': 80},
    'Barley': {'yield_per_acre': 20},
    'Oats': {'yield_per_acre': 18},
    'Sorghum': {'yield_per_acre': 15},
    'Millet': {'yield_per_acre': 12},
    'Soybeans': {'yield_per_acre': 10},
    'Canola': {'yield_per_acre': 15},
    'Lentils': {'yield_per_acre': 12},
    'Chickpeas': {'yield_per_acre': 10},
    'Coffee': {'yield_per_acre': 3},
    'Almonds': {'yield_per_acre': 2},
    'Walnuts': {'yield_per_acre': 3},
    'Pistachios': {'yield_per_acre': 2},
    'Potatoes': {'yield_per_acre': 250},
    'Onions': {'yield_per_acre': 200},
    'Tomatoes': {'yield_per_acre': 250},
    'Garlic': {'yield_per_acre': 80},
    'Cotton': {'yield_per_acre': 8},
    'Wool': {'yield_per_acre': 3},
    'Peanuts': {'yield_per_acre': 15}
}

update_crop_prices()

def create_pie_chart(crops, probabilities):
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=crops,
        values=probabilities,
        hole=.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Crop Distribution Analysis",
        title_x=0.5,
        title_font=dict(size=20, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(t=50, l=0, r=0, b=0),
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_revenue_chart(revenue_data):
    crops = [data['crop'].title() for data in revenue_data]
    revenues = [data['gross_revenue'] for data in revenue_data]
    
    fig = go.Figure(data=[
        go.Bar(
            x=crops,
            y=revenues,
            marker_color='#2ecc71'
        )
    ])
    
    fig.update_layout(
        title="Expected Revenue by Crop",
        title_x=0.5,
        xaxis_title="Crops",
        yaxis_title="Expected Revenue (₹)",
        title_font=dict(size=20, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(t=50, l=50, r=50, b=50),
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def calculate_revenue(crop_name):
    crop_name = crop_name.title()
    if crop_name in CROP_DATA:
        crop_info = CROP_DATA[crop_name]
        yield_per_acre = crop_info['yield_per_acre']
        price_per_kg = crop_info.get('price_per_kg', 0)
        gross_revenue = yield_per_acre * price_per_kg
        return {
            'crop': crop_name,
            'yield_per_acre': yield_per_acre,
            'price_per_kg': price_per_kg,
            'gross_revenue': gross_revenue,
            'last_updated': crop_info.get('last_updated', 'N/A')
        }
    return None

@app.route('/')
def home():
    update_crop_prices()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()

        input_data = {
            'Nitrogen': float(data['Nitrogen']),
            'Phosphorus': float(data['Phosphorus']),
            'Potassium': float(data['Potassium']),
            'Temperature': float(data['Temperature']),
            'Humidity': float(data['Humidity']),
            'pH': float(data['pH']),
            'Soil Type': data['Soil Type'],
            'Period of Month': data['Period of Month']
        }

        input_data['NPK_Ratio'] = (input_data['Nitrogen'] + input_data['Phosphorus'] + input_data['Potassium']) / 3
        input_data['Temp_Humidity_Index'] = input_data['Temperature'] * input_data['Humidity'] / 100
        input_data['Soil_Moisture_Index'] = input_data['Humidity'] * input_data['pH'] / 100
        input_data['NPK_TH_Index'] = input_data['NPK_Ratio'] * input_data['Temp_Humidity_Index'] / 100

        df = pd.DataFrame([input_data])
        df['Soil Type'] = label_encoder_soil.transform(df['Soil Type'])
        df['Period of Month'] = label_encoder_period.transform(df['Period of Month'])
        
        df = df[feature_names]
        
        scaled_features = scaler.transform(df)
        
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[0]
        
        top_3_idx = probabilities.argsort()[-3:][::-1]
        top_3_crops = label_encoder_crop.inverse_transform(top_3_idx)
        top_3_probs = probabilities[top_3_idx] * 100
        
        update_crop_prices()
        
        revenue_data = []
        for crop in top_3_crops:
            rev_data = calculate_revenue(crop)
            if rev_data:
                revenue_data.append(rev_data)
        
        pie_chart = create_pie_chart(top_3_crops, top_3_probs)
        revenue_chart = create_revenue_chart(revenue_data)
        
        result = {
            'predicted_crop': label_encoder_crop.inverse_transform(prediction)[0],
            'confidence': float(max(probabilities) * 100),
            'top_3_crops': [
                {'crop': crop, 'probability': prob} 
                for crop, prob in zip(top_3_crops, top_3_probs)
            ],
            'revenue_data': revenue_data,
            'pie_chart': pie_chart,
            'revenue_chart': revenue_chart,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        app.config['LAST_PREDICTION'] = result
        
        if request.is_json:
            return jsonify(result)
        return render_template('result.html', result=result)
        
    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template('index.html', error=error_msg)

@app.route('/yield-calculator')
def yield_calculator():
    update_crop_prices()
    last_prediction = app.config.get('LAST_PREDICTION', {})
    return render_template('yield_calculator.html', prediction=last_prediction)

@app.route('/calculate-profit', methods=['POST'])
def calculate_profit():
    try:
        costs = {
            'seed_cost': float(request.form['seed_cost']),
            'land_cost': float(request.form['land_cost']),
            'fertilizer_cost': float(request.form['fertilizer_cost']),
            'irrigation_cost': float(request.form['irrigation_cost']),
            'labor_cost': float(request.form['labor_cost'])
        }
        
        total_cost = sum(costs.values())
        last_prediction = app.config.get('LAST_PREDICTION', {})
        revenue_data = last_prediction.get('revenue_data', [])
        
        profits = []
        for crop_revenue in revenue_data:
            if crop_revenue:
                profit = {
                    'crop': crop_revenue['crop'],
                    'gross_revenue': crop_revenue['gross_revenue'],
                    'total_cost': total_cost,
                    'net_profit': crop_revenue['gross_revenue'] - total_cost,
                    'last_updated': crop_revenue.get('last_updated', 'N/A')
                }
                profits.append(profit)
        
        return render_template('profit_result.html', 
                             profits=profits, 
                             costs=costs,
                             calculation_date=datetime.now().strftime("%Y-%m-%d"))
                             
    except Exception as e:
        return render_template('yield_calculator.html', 
                             error=f"Calculation Error: {str(e)}",
                             prediction=app.config.get('LAST_PREDICTION', {}))

@app.route('/prices')
def show_prices():
    market_prices = load_latest_prices()
    return render_template('prices.html', prices=market_prices)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)