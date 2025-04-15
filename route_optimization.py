import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point

def optimize_routes(data_points, road_network=None, algorithm='Dijkstra'):
    """
    Perform route optimization between high-demand points
    
    Parameters:
    -----------
    data_points : GeoDataFrame
        Points with demand predictions
    road_network : GeoDataFrame, optional
        Road network data
    algorithm : str
        Routing algorithm to use
        
    Returns:
    --------
    GeoDataFrame
        Optimized routes
    """
    if data_points is None or len(data_points) == 0:
        return None
    
    # Get top demand points
    if 'future_demand' in data_points.columns:
        top_points = data_points.sort_values('future_demand', ascending=False).head(5)
    else:
        top_points = data_points.head(5)
    
    # Create a simple network graph
    G = nx.Graph()
    
    # Add nodes for each point
    for idx, row in top_points.iterrows():
        if hasattr(row, 'geometry') and isinstance(row.geometry, Point):
            G.add_node(idx, pos=(row.geometry.x, row.geometry.y))
    
    # Add edges between nodes (complete graph)
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            
            # Calculate Euclidean distance
            pos_i = G.nodes[node_i]['pos']
            pos_j = G.nodes[node_j]['pos']
            
            # Distance in degrees (approximate)
            dist = ((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)**0.5
            
            # Add edge with distance as weight
            G.add_edge(node_i, node_j, weight=dist)
    
    # Find optimal routes
    routes = []
    route_data = []
    
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            source = nodes[i]
            target = nodes[j]
            
            try:
                # Find shortest path
                if algorithm == 'Dijkstra':
                    path = nx.dijkstra_path(G, source, target, weight='weight')
                else:  # A* search
                    path = nx.astar_path(G, source, target, weight='weight')
                
                # Extract coordinates for the path
                coords = [G.nodes[node]['pos'] for node in path]
                
                # Create linestring
                line = LineString(coords)
                
                # Add to routes
                routes.append(line)
                route_data.append({
                    'source': source,
                    'target': target,
                    'distance': nx.dijkstra_path_length(G, source, target, weight='weight')
                })
            except Exception as e:
                print(f"Error finding path: {e}")
    
    # Create GeoDataFrame with routes
    if routes:
        routes_gdf = gpd.GeoDataFrame(route_data, geometry=routes, crs="EPSG:4326")
        return routes_gdf
    else:
        return None

def calculate_route_statistics(routes_data):
    """
    Calculate statistics for the optimized routes
    
    Parameters:
    -----------
    routes_data : GeoDataFrame
        Optimized routes
        
    Returns:
    --------
    dict
        Dictionary with route statistics
    """
    if routes_data is None or len(routes_data) == 0:
        return {}
    
    # Number of routes
    num_routes = len(routes_data)
    
    # Convert distances to approximate km (at Calgary's latitude)
    distances_km = [row.get('distance', 0) * 111 for _, row in routes_data.iterrows()]
    
    # Calculate statistics
    avg_distance = sum(distances_km) / len(distances_km) if distances_km else 0
    total_distance = sum(distances_km)
    max_distance = max(distances_km) if distances_km else 0
    min_distance = min(distances_km) if distances_km else 0
    
    # Create distance distribution data
    distance_data = pd.DataFrame({
        'Route': [f"Route {i+1}" for i in range(len(distances_km))],
        'Distance (km)': distances_km
    }).sort_values('Distance (km)', ascending=False)
    
    return {
        'num_routes': num_routes,
        'avg_distance': avg_distance,
        'total_distance': total_distance,
        'max_distance': max_distance,
        'min_distance': min_distance,
        'distance_data': distance_data
    }