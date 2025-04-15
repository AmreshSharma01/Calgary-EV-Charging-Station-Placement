import pandas as pd
import streamlit as st

def format_recommendations_table(recommendations):
    """
    Format recommendations dataframe for display
    
    Parameters:
    -----------
    recommendations : DataFrame
        Raw recommendations data
        
    Returns:
    --------
    DataFrame
        Formatted recommendations table for display
    """
    if recommendations is None or len(recommendations) == 0:
        return None
    
    # Create a copy to avoid modifying the original
    display_df = recommendations.copy()
    
    # Add rank column
    display_df['rank'] = range(1, len(display_df) + 1)
    
    # Round scores for better display
    if 'combined_score' in display_df.columns:
        display_df['score'] = display_df['combined_score'].round(2)
    elif 'importance_score' in display_df.columns:
        display_df['score'] = display_df['importance_score'].round(2)
    
    # Select and rename columns for display
    if 'nearest_community' in display_df.columns:
        display_cols = ['rank', 'nearest_community', 'community_class', 'score', 'future_demand']
        rename_map = {
            'rank': 'Rank',
            'nearest_community': 'Community',
            'community_class': 'Type',
            'score': 'Score',
            'future_demand': 'Demand'
        }
    else:
        display_cols = ['rank', 'cluster_id', 'score', 'future_demand', 'source']
        rename_map = {
            'rank': 'Rank',
            'cluster_id': 'Cluster',
            'score': 'Score',
            'future_demand': 'Demand',
            'source': 'Source'
        }
    
    # Select only columns that exist in the dataframe
    valid_cols = [col for col in display_cols if col in display_df.columns]
    clean_display = display_df[valid_cols].rename(columns=rename_map)
    
    return clean_display

def calculate_stations_per_100k(num_stations, population):
    """
    Calculate the number of charging stations per 100,000 people
    
    Parameters:
    -----------
    num_stations : int
        Number of charging stations
    population : int
        Total population
        
    Returns:
    --------
    float
        Stations per 100,000 people
    """
    if population <= 0:
        return 0
    
    return (num_stations / population) * 100000

def get_growth_rate_description(growth_rate):
    """
    Get a description of the growth rate
    
    Parameters:
    -----------
    growth_rate : float
        Monthly growth rate
        
    Returns:
    --------
    str
        Description of the growth rate
    """
    annual_rate = (1 + growth_rate)**12 - 1
    
    if annual_rate < 0.1:
        return "modest"
    elif annual_rate < 0.25:
        return "moderate"
    elif annual_rate < 0.5:
        return "strong"
    else:
        return "rapid"

def get_feature_info():
    """
    Get information about which features to use for demand prediction based on available data
    
    Returns:
    --------
    list
        List of feature names to use for prediction
    """
    # Default features to look for
    default_features = [
        'Volume',
        'weight',
        'num_chargers',
        'CLASS_CODE',
        'traffic_volume',
        'importance_score'
    ]
    
    return default_features