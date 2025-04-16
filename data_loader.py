import pandas as pd
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point, LineString, MultiLineString
import re
import numpy as np

def load_csv_with_geometry(file_path, geometry_column=None, x_column=None, y_column=None, encoding='utf-8'):
    """
    Load CSV file and convert to GeoDataFrame with geometry
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    geometry_column : str
        Name of column containing geometry information (POINT, LINESTRING, etc.)
    x_column : str
        Name of column containing x coordinate (longitude)
    y_column : str
        Name of column containing y coordinate (latitude)
    encoding : str
        File encoding
        
    Returns:
    --------
    GeoDataFrame
        Loaded and processed data
    """
    # Try to load the data
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        
        # Clean the data - handle NaN values
        # Replace NaN values in numeric columns with appropriate defaults
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(0)
            elif df[col].dtype == object:
                df[col] = df[col].fillna('')
        
        # Check if we have geometry column
        if geometry_column and geometry_column in df.columns:
            # Parse geometry from WKT
            # Replace the parse_wkt function in data_loader.py with this version:

            def parse_wkt(wkt_str):
                """Parse Well-Known Text (WKT) geometry string safely."""
                # Check if input is None or NaN
                if wkt_str is None or pd.isna(wkt_str):
                    return None

                # Ensure we're working with a string
                if not isinstance(wkt_str, str):
                    try:
                        wkt_str = str(wkt_str)
                    except:
                        return None

                try:
                    if wkt_str.startswith('POINT'):
                        # Extract coordinates using regex
                        coords = re.findall(r'-?\d+\.\d+', wkt_str)
                        if len(coords) >= 2:
                            return Point(float(coords[0]), float(coords[1]))
                    elif wkt_str.startswith('LINESTRING'):
                        # Extract all coordinates
                        coords_str = wkt_str.replace('LINESTRING (', '').replace(')', '')
                        # Only split if we have a string
                        if isinstance(coords_str, str):
                            coords_pairs = coords_str.split(', ')
                            # Only process if we have valid pairs
                            if coords_pairs and all(isinstance(pair, str) for pair in coords_pairs):
                                coords = []
                                for pair in coords_pairs:
                                    if ' ' in pair:  # Only split if there's a space
                                        parts = pair.split()
                                        if len(parts) >= 2:
                                            try:
                                                x, y = float(parts[0]), float(parts[1])
                                                coords.append((x, y))
                                            except (ValueError, TypeError):
                                                continue
                                if coords:
                                    return LineString(coords)
                    elif wkt_str.startswith('MULTILINESTRING'):
                        # This would require more complex parsing
                        # For now, we'll return None for multilinestrings
                        return None
                except Exception as e:
                    print(f"Error parsing WKT: {e}")
                    return None
                return None
            
            # Apply parsing to geometry column
            geometries = df[geometry_column].apply(parse_wkt)
            valid_geometries = geometries.dropna()
            
            if len(valid_geometries) > 0:
                gdf = gpd.GeoDataFrame(df.loc[valid_geometries.index], geometry=valid_geometries, crs="EPSG:4326")
            else:
                # No valid geometries found, return regular DataFrame
                return df
            
        # Check if we have x and y columns
        elif x_column and y_column and x_column in df.columns and y_column in df.columns:
            # Create Point geometries from coordinates
            geometries = []
            valid_indices = []
            
            for idx, row in df.iterrows():
                try:
                    # Handle NaN or invalid values
                    if pd.isna(row[x_column]) or pd.isna(row[y_column]):
                        continue
                        
                    x, y = float(row[x_column]), float(row[y_column])
                    geometries.append(Point(x, y))
                    valid_indices.append(idx)
                except (ValueError, TypeError):
                    # Skip invalid coordinates
                    continue
            
            if geometries:
                gdf = gpd.GeoDataFrame(df.iloc[valid_indices].reset_index(drop=True), 
                                      geometry=geometries, crs="EPSG:4326")
            else:
                # No valid geometries, return regular DataFrame
                return df
        else:
            # No geometry information, return regular DataFrame
            return df
        
        return gdf
    
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {str(e)}")
        return None

@st.cache_data
def load_all_datasets(data_dir="data/"):
    """
    Load all datasets for the application
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
        
    Returns:
    --------
    tuple
        Tuple containing all loaded datasets
    """
    # Load EV stations
    ev_stations = load_csv_with_geometry(f"{data_dir}calgary_ev_stations.csv", 
                                         x_column='longitude', 
                                         y_column='latitude')
    
    # Load traffic volumes
    traffic_volumes = load_csv_with_geometry(f"{data_dir}Traffic_Volumes_for_2023.csv",
                                            geometry_column='multilinestring')
    
    # Load community points
    community_points = load_csv_with_geometry(f"{data_dir}Community_Points_20250414.csv",
                                             x_column='longitude',
                                             y_column='latitude')
    
    # Load population data - try the new file first, then fall back to old filename
    try:
        population_data = pd.read_csv(f"{data_dir}calgary population dataset new.csv", encoding='utf-8')
    except:
        try:
            # Try with the old filename if the new one doesn't exist
            population_data = pd.read_csv(f"{data_dir}calgary population dataset.csv", encoding='utf-8')
        except:
            try:
                # Try with different encoding if utf-8 fails
                population_data = pd.read_csv(f"{data_dir}calgary population dataset new.csv", encoding='latin1')
            except Exception as e:
                st.warning(f"Could not load population dataset: {str(e)}")
                population_data = None
        
    # Load road network
    road_network = load_csv_with_geometry(f"{data_dir}Major_Road_Network_2025.csv",
                                         geometry_column='the_geom')
    
    return ev_stations, traffic_volumes, community_points, population_data, road_network

def prepare_combined_data(ev_stations, traffic_volumes, community_points):
    """
    Combine data from different sources for analysis
    
    Parameters:
    -----------
    ev_stations : GeoDataFrame
        EV charging stations data
    traffic_volumes : GeoDataFrame
        Traffic volume data
    community_points : GeoDataFrame
        Community points data
        
    Returns:
    --------
    GeoDataFrame
        Combined data with relevant attributes
    """
    data_points = []
    
    # Add EV stations
    if ev_stations is not None and isinstance(ev_stations, gpd.GeoDataFrame):
        ev_points = ev_stations.copy()
        ev_points['source'] = 'ev_station'
        
        # Ensure num_chargers is numeric and handle NaN values
        if 'num_chargers' in ev_points.columns:
            ev_points['num_chargers'] = pd.to_numeric(ev_points['num_chargers'], errors='coerce').fillna(1)
        
        ev_points['weight'] = ev_points.get('num_chargers', 1)
        data_points.append(ev_points)
    
    # Add community points
    if community_points is not None and isinstance(community_points, gpd.GeoDataFrame):
        # Filter to include only residential communities for better analysis
        if 'CLASS' in community_points.columns:
            residential = community_points[community_points['CLASS'] == 'Residential'].copy()
        else:
            residential = community_points.copy()
            
        residential['source'] = 'community'
        
        # Ensure CLASS_CODE is numeric
        if 'CLASS_CODE' in residential.columns:
            residential['CLASS_CODE'] = pd.to_numeric(residential['CLASS_CODE'], errors='coerce').fillna(0)
        
        # Weight based on structure (newer communities might have higher EV adoption)
        def get_weight(structure):
            # Handle None, NaN, or non-string values
            if pd.isna(structure):
                return 1
                
            # Convert to string safely and check
            try:
                # Only process if it's a string or can be converted to a string
                structure_str = str(structure).lower()
                if '2010s' in structure_str:
                    return 3
                elif '2000s' in structure_str:
                    return 2
            except:
                # If any error occurs during conversion or processing, return default
                return 1
                
            # Default return
            return 1
        
        if 'COMM_STRUCTURE' in residential.columns:
            residential['weight'] = residential['COMM_STRUCTURE'].apply(get_weight)
        else:
            residential['weight'] = 1
            
        data_points.append(residential)
    
    # Add traffic points (if we can extract points from it)
    if traffic_volumes is not None and isinstance(traffic_volumes, gpd.GeoDataFrame):
        # For traffic data, we'll try to extract points for analysis
        try:
            # Get centroid of each traffic segment
            traffic_points = traffic_volumes.copy()
            
            # Ensure Volume is numeric
            if 'Volume' in traffic_points.columns:
                traffic_points['Volume'] = pd.to_numeric(traffic_points['Volume'], errors='coerce').fillna(0)
            
            # Convert any multilinestring columns to actual geometries if needed
            if 'multilinestring' in traffic_points.columns and not isinstance(traffic_points.geometry[0], (Point, LineString)):
                # Create a new geometry column from the multilinestring text
                valid_indices = []
                new_geometries = []
                
                for idx, row in traffic_points.iterrows():
                    try:
                        if isinstance(row['multilinestring'], str):
                            geom = parse_wkt(row['multilinestring'])
                            if geom:
                                new_geometries.append(geom.centroid)
                                valid_indices.append(idx)
                    except:
                        continue
                
                if valid_indices:
                    traffic_points = traffic_points.loc[valid_indices].copy()
                    traffic_points['geometry'] = new_geometries
                else:
                    # No valid geometries extracted
                    return None
            elif hasattr(traffic_points, 'geometry'):
                # Process existing geometries to ensure they're points
                valid_indices = []
                new_geometries = []
                
                for idx, row in traffic_points.iterrows():
                    try:
                        if row.geometry is not None:
                            if hasattr(row.geometry, 'centroid'):
                                new_geometries.append(row.geometry.centroid)
                                valid_indices.append(idx)
                    except:
                        continue
                
                if valid_indices:
                    traffic_points = traffic_points.loc[valid_indices].copy()
                    traffic_points['geometry'] = new_geometries
            
            traffic_points['source'] = 'traffic'
            traffic_points['weight'] = 1.0
            
            # Set weight based on volume if available
            if 'Volume' in traffic_points.columns:
                for idx, row in traffic_points.iterrows():
                    try:
                        volume = float(row['Volume'])
                        traffic_points.at[idx, 'weight'] = volume / 1000
                    except:
                        pass
                    
            data_points.append(traffic_points)
        except Exception as e:
            st.warning(f"Could not process traffic data: {str(e)}")
    
    # Combine all data points
    if data_points:
        try:
            # Combine and reset index
            combined = pd.concat(data_points, ignore_index=True)
            
            # Convert to GeoDataFrame if it's not already
            if not isinstance(combined, gpd.GeoDataFrame):
                combined = gpd.GeoDataFrame(combined, geometry='geometry', crs="EPSG:4326")
            
            # Final cleaning - handle any remaining NaN values
            for col in combined.columns:
                if col != 'geometry' and combined[col].dtype in [np.float64, np.int64]:
                    combined[col] = combined[col].fillna(0)
                elif col != 'geometry' and combined[col].dtype == object:
                    combined[col] = combined[col].fillna('')
            
            # Filter out rows with None or invalid geometries
            combined = combined[combined.geometry.notna()]
            
            return combined
            
        except Exception as e:
            st.error(f"Error combining data points: {str(e)}")
            return None
    else:
        return None