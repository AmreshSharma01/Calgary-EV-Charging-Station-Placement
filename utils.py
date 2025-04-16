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
    
    # Format income factor if available
    if 'income_factor' in display_df.columns:
        display_df['income_influence'] = display_df['income_factor'].round(2)
    
    # Select and rename columns for display
    if 'nearest_community' in display_df.columns:
        display_cols = ['rank', 'nearest_community', 'community_class', 'score', 'future_demand']
        # Add income influence if available
        if 'income_influence' in display_df.columns:
            display_cols.append('income_influence')
            
        rename_map = {
            'rank': 'Rank',
            'nearest_community': 'Community',
            'community_class': 'Type',
            'score': 'Score',
            'future_demand': 'Demand',
            'income_influence': 'Income Factor'
        }
    else:
        display_cols = ['rank', 'cluster_id', 'score', 'future_demand', 'source']
        # Add income influence if available
        if 'income_influence' in display_df.columns:
            display_cols.append('income_influence')
            
        rename_map = {
            'rank': 'Rank',
            'cluster_id': 'Cluster',
            'score': 'Score',
            'future_demand': 'Demand',
            'source': 'Source',
            'income_influence': 'Income Factor'
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

def extract_household_income(population_data):
    """
    Extract household income data from population dataset
    
    Parameters:
    -----------
    population_data : DataFrame
        Population data
        
    Returns:
    --------
    dict
        Dictionary with household income metrics
    """
    results = {
        'avg_income': 129000,  # Default from 2021 census
        'median_income': 98000  # Default from 2021 census
    }
    
    if population_data is None:
        return results
    
    try:
        # Look for average income in the dataset
        avg_income_rows = population_data[population_data['Characteristic'].str.contains('Average total income of household', case=False, na=False)]
        if len(avg_income_rows) > 0:
            value_col = 'Value' if 'Value' in avg_income_rows.columns else 'Total'
            avg_value = avg_income_rows.iloc[0][value_col]
            if isinstance(avg_value, (int, float)) or (isinstance(avg_value, str) and str(avg_value).replace(',', '').replace('$', '').strip().isdigit()):
                results['avg_income'] = int(float(str(avg_value).replace(',', '').replace('$', '').strip()))
        
        # Look for median income in the dataset
        median_income_rows = population_data[population_data['Characteristic'].str.contains('Median total income of household', case=False, na=False)]
        if len(median_income_rows) > 0:
            value_col = 'Value' if 'Value' in median_income_rows.columns else 'Total'
            median_value = median_income_rows.iloc[0][value_col]
            if isinstance(median_value, (int, float)) or (isinstance(median_value, str) and str(median_value).replace(',', '').replace('$', '').strip().isdigit()):
                results['median_income'] = int(float(str(median_value).replace(',', '').replace('$', '').strip()))
    except Exception as e:
        print(f"Error extracting income data: {e}")
    
    return results

def calculate_income_factor(income):
    """
    Calculate income factor for EV adoption prediction
    
    Parameters:
    -----------
    income : float
        Average household income
        
    Returns:
    --------
    float
        Income factor (1.0 = average, higher values indicate higher adoption)
    """
    # Normalize income to a reasonable scale for EV adoption
    # Base 1.0 at $100,000 income level
    return income / 100000

# Add this function to utils.py
# This will help diagnose and fix any issues with datatypes

def debug_clustered_data(clustered_data):
    """
    Debug and fix datatype issues in clustered data
    
    Parameters:
    -----------
    clustered_data : GeoDataFrame
        Clustered data that may have datatype issues
        
    Returns:
    --------
    GeoDataFrame
        Fixed data with proper datatypes
    dict
        Diagnostics information
    """
    if clustered_data is None:
        return None, {"error": "No data provided"}
    
    diagnostics = {
        "original_columns": list(clustered_data.columns),
        "column_types": {},
        "problematic_columns": [],
        "fixed_columns": []
    }
    
    # First, check all column types
    for col in clustered_data.columns:
        if col == 'geometry':
            continue
            
        col_type = str(clustered_data[col].dtype)
        diagnostics["column_types"][col] = col_type
        
        # Check for any columns that should be numeric but aren't
        if col in ['importance_score', 'cluster', 'weight', 'current_demand', 'future_demand']:
            try:
                # Try to convert to numeric
                numeric_values = pd.to_numeric(clustered_data[col], errors='coerce')
                
                # Check if we have any non-numeric values
                if numeric_values.isna().any() and not clustered_data[col].isna().any():
                    diagnostics["problematic_columns"].append(col)
                    
                    # Fix this column
                    clustered_data[col] = numeric_values
                    diagnostics["fixed_columns"].append(col)
            except:
                diagnostics["problematic_columns"].append(col)
                
    # Fix importance_score if it's a problematic column
    if 'importance_score' in clustered_data.columns and 'importance_score' in diagnostics["problematic_columns"]:
        # Create a new importance score based on other columns
        try:
            # Set a default
            clustered_data['importance_score'] = 1.0
            
            # Enhance based on source type
            if 'source' in clustered_data.columns:
                # Higher importance for EV stations
                clustered_data.loc[clustered_data['source'] == 'ev_station', 'importance_score'] = 2.0
                
                # Higher importance for traffic with high volume
                if 'Volume' in clustered_data.columns:
                    high_volume = clustered_data['Volume'] > clustered_data['Volume'].median()
                    traffic_mask = (clustered_data['source'] == 'traffic') & high_volume
                    clustered_data.loc[traffic_mask, 'importance_score'] = 1.5
                    
                # Higher importance for residential communities
                if 'CLASS' in clustered_data.columns:
                    residential_mask = (clustered_data['source'] == 'community') & (clustered_data['CLASS'] == 'Residential')
                    clustered_data.loc[residential_mask, 'importance_score'] = 1.8
            
            diagnostics["fixed_columns"].append('importance_score')
        except:
            # If we can't fix it, just set it to 1.0
            clustered_data['importance_score'] = 1.0
            
    return clustered_data, diagnostics