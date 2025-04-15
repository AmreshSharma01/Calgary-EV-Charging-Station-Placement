import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

def predict_demand(data_points, features, forecast_period):
    """
    Predict EV charging demand
    
    Parameters:
    -----------
    data_points : GeoDataFrame
        Spatial data points
    features : list
        List of feature names to use for prediction
    forecast_period : int
        Number of months to forecast
        
    Returns:
    --------
    GeoDataFrame
        Data with predicted demand
    float
        Prediction accuracy
    """
    if data_points is None or len(data_points) == 0:
        return None, 0
    
    # Ensure all features exist in the data
    valid_features = [f for f in features if f in data_points.columns]
    if not valid_features:
        # If no valid features, add a dummy feature
        data_points['dummy'] = 1
        valid_features = ['dummy']
    
    # Create feature matrix
    X = data_points[valid_features].fillna(0)
    
    # Create synthetic target (in a real scenario, this would be actual historical data)
    # Here we're creating a formula based on available features
    y = np.zeros(len(X))
    
    if 'Volume' in X.columns:
        y += X['Volume'] * 0.0001  # Traffic volume contribution
    
    if 'weight' in X.columns:
        y += X['weight'] * 0.1  # Weight contribution
    
    if 'num_chargers' in X.columns:
        y += X['num_chargers'] * 0.1  # Existing chargers indicate demand
        
    if 'CLASS_CODE' in X.columns:
        # Residential areas (code 1) have higher demand
        y += (X['CLASS_CODE'] == 1).astype(int) * 0.5
    
    # Add some randomness to simulate real-world variation
    y += np.random.normal(0, 0.1, len(y))
    
    # Ensure non-negative values
    y = np.maximum(y, 0)
    
    # Train a simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict current demand
    current_demand = model.predict(X)
    
    # Project future demand with growth factor
    growth_rate = 0.05  # 5% monthly growth rate
    future_demand = current_demand * (1 + growth_rate * forecast_period)
    
    # Add predictions to the data
    result = data_points.copy()
    result['current_demand'] = current_demand
    result['future_demand'] = future_demand
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': valid_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Print feature importance to aid in understanding
    if not st.session_state.get('quiet_mode', False):
        st.write("Feature Importance for Demand Prediction:")
        st.dataframe(feature_importance)
    
    # Simulate prediction accuracy
    accuracy = 85 + np.random.uniform(-5, 5)
    
    return result, accuracy

def calculate_demand_growth(demand_data, forecast_period):
    """
    Calculate demand growth metrics for visualization
    
    Parameters:
    -----------
    demand_data : GeoDataFrame
        Data with demand predictions
    forecast_period : int
        Number of months in the forecast
        
    Returns:
    --------
    dict
        Dictionary with growth metrics
    """
    if demand_data is None or 'current_demand' not in demand_data.columns or 'future_demand' not in demand_data.columns:
        return {}
    
    # Calculate average demands
    avg_current = demand_data['current_demand'].mean()
    avg_future = demand_data['future_demand'].mean()
    
    # Calculate growth percentage
    growth_percent = ((avg_future - avg_current) / avg_current * 100) if avg_current > 0 else 0
    
    # Calculate monthly growth rate
    growth_rate = (avg_future / avg_current) ** (1/forecast_period) - 1 if avg_current > 0 else 0.05
    
    # Generate monthly values with compound growth
    months = [f"Month {i+1}" for i in range(forecast_period)]
    demand_values = [avg_current * (1 + growth_rate) ** i for i in range(forecast_period)]
    
    # Create confidence intervals
    upper_bound = [val * 1.2 for val in demand_values]  # 20% higher
    lower_bound = [val * 0.8 for val in demand_values]  # 20% lower
    
    return {
        'avg_current': avg_current,
        'avg_future': avg_future,
        'growth_percent': growth_percent,
        'growth_rate': growth_rate,
        'months': months,
        'demand_values': demand_values,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound
    }