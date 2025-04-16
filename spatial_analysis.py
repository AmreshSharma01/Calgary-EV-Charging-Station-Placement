import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

def perform_clustering(data_points, method="DBSCAN", eps=1.0, min_samples=5, n_clusters=8):
    """
    Perform clustering on spatial data points
    
    Parameters:
    -----------
    data_points : GeoDataFrame
        Spatial data points to cluster
    method : str
        Clustering method ("DBSCAN" or "K-Means")
    eps : float
        Epsilon parameter for DBSCAN (in km)
    min_samples : int
        Minimum samples parameter for DBSCAN
    n_clusters : int
        Number of clusters for K-Means
        
    Returns:
    --------
    GeoDataFrame
        Clustered data with cluster labels
    """
    if data_points is None or len(data_points) == 0:
        return None
    
    # Extract coordinates from geometry
    X = []
    valid_indices = []
    
    try:
        for i, p in enumerate(data_points.geometry):
            try:
                if p is not None and hasattr(p, 'x') and hasattr(p, 'y'):
                    # Check if x and y are valid numbers
                    if not pd.isna(p.x) and not pd.isna(p.y):
                        X.append([p.x, p.y])
                        valid_indices.append(i)
            except Exception as e:
                print(f"Skipping invalid geometry at index {i}: {e}")
    except Exception as e:
        st.error(f"Error extracting coordinates: {str(e)}")
        return None
    
    # If no valid points, return None
    if not X or len(X) < 2:  # Need at least 2 points for clustering
        st.warning("Not enough valid geometries for clustering analysis.")
        return None
    
    # Convert to numpy array
    X = np.array(X)
    
    # Create a subset of the dataframe with valid geometries
    valid_data = data_points.iloc[valid_indices].copy().reset_index(drop=True)
    
    try:
        # Scale the data
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply clustering
        if method == "DBSCAN":
            # Convert distance-based epsilon from km to scaled units
            # Approximate conversion based on Calgary's latitude
            km_to_degree = 0.009  # Approximate for Calgary's latitude
            eps_scaled = eps * km_to_degree / max(np.std(X[:, 0]), 0.0001)  # Avoid division by zero
            
            # Apply DBSCAN clustering
            cluster_model = DBSCAN(eps=eps_scaled, min_samples=min_samples)
            labels = cluster_model.fit_predict(X_scaled)
        else:  # K-Means
            # Ensure n_clusters is not greater than number of samples
            n_clusters = min(n_clusters, len(X))
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = cluster_model.fit_predict(X_scaled)
        
        # Add cluster labels to the data
        valid_data['cluster'] = labels
        
    except Exception as e:
        st.error(f"Error during clustering: {str(e)}")
        return None
    
    # Calculate importance score based on available attributes
    try:
        def calculate_importance(row):
            score = 1.0
            
            # If it's from EV stations data
            if 'num_chargers' in valid_data.columns:
                try:
                    num_chargers = pd.to_numeric(row.get('num_chargers', 1), errors='coerce')
                    if not pd.isna(num_chargers):
                        score *= (1 + num_chargers / 10)
                except:
                    pass
            
            # If it's from traffic data
            if 'Volume' in valid_data.columns:
                try:
                    volume = pd.to_numeric(row.get('Volume', 1000), errors='coerce')
                    if not pd.isna(volume):
                        score *= (1 + volume / 50000)
                except:
                    pass
                
            # If it's from community data
            if 'CLASS_CODE' in valid_data.columns:
                try:
                    # Higher importance for residential areas (class code 1)
                    class_code = pd.to_numeric(row.get('CLASS_CODE', 0), errors='coerce')
                    if not pd.isna(class_code) and class_code == 1:
                        score *= 1.5
                except:
                    pass
                    
            return min(score, 5)  # Cap at 5
        
        valid_data['importance_score'] = valid_data.apply(calculate_importance, axis=1)
    except Exception as e:
        st.warning(f"Error calculating importance scores: {str(e)}")
        # Add default importance scores
        valid_data['importance_score'] = 1.0
    
    return valid_data

def generate_recommendations(clustered_data, demand_data, community_points=None):
    """
    Generate recommendations for EV charging station placement
    
    Parameters:
    -----------
    clustered_data : GeoDataFrame
        Clustering results with cluster labels
    demand_data : GeoDataFrame
        Demand prediction results
    community_points : GeoDataFrame, optional
        Community points data for reference
        
    Returns:
    --------
    DataFrame
        Ranked recommendations for station placement
    """
    if clustered_data is None or demand_data is None:
        return None
    
    try:
        # Get top points from each valid cluster based on demand
        valid_clusters = clustered_data[clustered_data['cluster'] >= 0]
        
        # Create an empty list to store recommendations
        recommendations = []
        
        # Process each cluster
        for cluster_id in valid_clusters['cluster'].unique():
            try:
                # Get points in this cluster
                cluster_points = valid_clusters[valid_clusters['cluster'] == cluster_id]
                
                # Since we're working with the same base data, find points with maximum importance
                if 'importance_score' in cluster_points.columns and 'future_demand' in demand_data.columns:
                    # Sort by importance and demand
                    sorted_points = cluster_points.sort_values('importance_score', ascending=False)
                    
                    if len(sorted_points) > 0:
                        top_point = sorted_points.iloc[0]
                        
                        # Find corresponding demand
                        # Simple approach using geometry proximity
                        distances = [top_point.geometry.distance(p.geometry) for _, p in demand_data.iterrows()]
                        closest_idx = np.argmin(distances)
                        closest_demand = demand_data.iloc[closest_idx]
                        
                        # Get income factor if available
                        income_factor = closest_demand.get('income_factor', 1.0)
                        
                        # Create recommendation
                        rec = {
                            'cluster_id': int(cluster_id),
                            'latitude': float(top_point.geometry.y),
                            'longitude': float(top_point.geometry.x),
                            'importance_score': float(top_point.get('importance_score', 0)),
                            'future_demand': float(closest_demand.get('future_demand', 0)),
                            'income_factor': float(income_factor),
                            'source': str(top_point.get('source', 'Unknown'))
                        }
                        
                        # Add closest community name if it's a community point
                        if community_points is not None:
                            try:
                                # Find closest community
                                comm_distances = [top_point.geometry.distance(p.geometry) 
                                                for _, p in community_points.iterrows()]
                                closest_comm_idx = np.argmin(comm_distances)
                                closest_comm = community_points.iloc[closest_comm_idx]
                                
                                if 'NAME' in closest_comm:
                                    rec['nearest_community'] = str(closest_comm.get('NAME', 'Unknown'))
                                
                                if 'CLASS' in closest_comm:
                                    rec['community_class'] = str(closest_comm.get('CLASS', 'Unknown'))
                            except Exception as e:
                                print(f"Error adding community info: {e}")
                        
                        recommendations.append(rec)
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {e}")
                continue
        
        # Create a DataFrame with recommendations
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            
            # Sort by importance, demand and income factor
            # Weight income factor slightly to prioritize areas with higher income
            rec_df['combined_score'] = rec_df['importance_score'] * rec_df['future_demand'] * (1 + 0.2 * (rec_df['income_factor'] - 1))
            rec_df = rec_df.sort_values('combined_score', ascending=False)
            
            return rec_df
        else:
            return None
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None