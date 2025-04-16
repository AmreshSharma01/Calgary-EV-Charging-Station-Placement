import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point

def create_folium_map(data, layer_name, color="blue"):
    """
    Create a Folium map with the provided geospatial data
    
    Parameters:
    -----------
    data : GeoDataFrame
        Geospatial data to be plotted
    layer_name : str
        Name of the layer for the legend
    color : str
        Color of the markers or lines
        
    Returns:
    --------
    folium.Map
        Folium map with the data plotted
    """
    # Check if data is empty
    if data is None or len(data) == 0:
        # Create a default map centered on Calgary
        m = folium.Map(location=[51.05, -114.07], zoom_start=11)
        return m
    
    # Create a map centered on the data if it has geometry
    if isinstance(data, gpd.GeoDataFrame) and not data.geometry.is_empty.all():
        center_lat = data.geometry.centroid.y.mean()
        center_lon = data.geometry.centroid.x.mean()
    else:
        # Default to Calgary downtown
        center_lat = 51.05
        center_lon = -114.07
        
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add the data to the map if it's a GeoDataFrame
    if isinstance(data, gpd.GeoDataFrame):
        for idx, row in data.iterrows():
            if hasattr(row, 'geometry') and row.geometry:
                if isinstance(row.geometry, Point):
                    # Create tooltip text based on available columns
                    tooltip_text = f"{layer_name} {idx}"
                    
                    # Add more details to tooltip if available
                    if hasattr(row, 'station_name'):
                        tooltip_text = f"{row.station_name}"
                    elif hasattr(row, 'NAME'):
                        tooltip_text = f"{row.NAME}"
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        tooltip=tooltip_text
                    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_cluster_map(clustered_data, ev_stations=None):
    """
    Create a map showing clustering results
    
    Parameters:
    -----------
    clustered_data : GeoDataFrame
        Clustering results with cluster labels
    ev_stations : GeoDataFrame, optional
        Existing EV stations for reference
        
    Returns:
    --------
    folium.Map
        Folium map with clustering results
    """
    # Create a map centered on Calgary
    m = folium.Map(location=[51.05, -114.07], zoom_start=11)
    
    # Add cluster points to the map
    if clustered_data is not None and 'cluster' in clustered_data.columns:
        for idx, row in clustered_data.iterrows():
            if row['cluster'] == -1:  # Noise points (DBSCAN only)
                color = 'gray'
                radius = 2
            else:
                # Generate a color based on cluster
                color = f"#{hash(str(row['cluster'])) % 0xFFFFFF:06x}"
                # Size based on importance
                radius = 4 + int(row.get('importance_score', 1) * 3)
            
            # Create popup content
            popup_content = f"""
            <b>Cluster:</b> {row['cluster'] if row['cluster'] >= 0 else 'Noise'}<br>
            <b>Source:</b> {row.get('source', 'Unknown')}<br>
            <b>Importance:</b> {row.get('importance_score', 0):.2f}
            """
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=popup_content
            ).add_to(m)
    
    # Add existing charging stations for reference
    if ev_stations is not None:
        for idx, row in ev_stations.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.7,
                tooltip=f"Existing Station: {row.get('station_name', 'Unknown')}"
            ).add_to(m)
    
    return m

def create_demand_map(demand_data, ev_stations=None):
    """
    Create a map showing demand predictions
    
    Parameters:
    -----------
    demand_data : GeoDataFrame
        Demand prediction results
    ev_stations : GeoDataFrame, optional
        Existing EV stations for reference
        
    Returns:
    --------
    folium.Map
        Folium map with demand predictions
    """
    # Create a map centered on Calgary
    m = folium.Map(location=[51.05, -114.07], zoom_start=11)
    
    # Add demand points to the map, sized by predicted demand
    if demand_data is not None and 'future_demand' in demand_data.columns:
        for idx, row in demand_data.iterrows():
            # Skip points with no demand
            if row.get('future_demand', 0) <= 0:
                continue
                
            # Scale radius by demand
            radius = 3 + int(row.get('future_demand', 1) * 5)
            # Limit radius to reasonable size
            radius = min(radius, 15)
            
            # Create color based on demand level (higher demand = more red)
            demand_level = min(row.get('future_demand', 0) / 3, 1)  # Normalize to 0-1
            color = f"rgb({int(255 * demand_level)}, {int(255 * (1-demand_level))}, 0)"
            
            # Create popup content
            popup_content = f"""
            <b>Current Demand:</b> {row.get('current_demand', 0):.2f}<br>
            <b>Future Demand:</b> {row.get('future_demand', 0):.2f}<br>
            <b>Source:</b> {row.get('source', 'Unknown')}
            """
            
            # Add income factor if available
            if 'income_factor' in row:
                popup_content += f"<br><b>Income Factor:</b> {row.get('income_factor', 1):.2f}"
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=popup_content
            ).add_to(m)
    
    # Add existing charging stations for reference
    if ev_stations is not None:
        for idx, row in ev_stations.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.7,
                tooltip=f"Existing Station: {row.get('station_name', 'Unknown')}"
            ).add_to(m)
    
    return m

def create_routes_map(routes_data, demand_data=None, ev_stations=None):
    """
    Create a map showing optimized routes
    
    Parameters:
    -----------
    routes_data : GeoDataFrame
        Route optimization results
    demand_data : GeoDataFrame, optional
        Demand prediction results to show high-demand points
    ev_stations : GeoDataFrame, optional
        Existing EV stations for reference
        
    Returns:
    --------
    folium.Map
        Folium map with routes
    """
    # Create a map centered on Calgary
    m = folium.Map(location=[51.05, -114.07], zoom_start=11)
    
    # Add routes to the map
    if routes_data is not None:
        for idx, row in routes_data.iterrows():
            # Create popup content
            popup_content = f"""
            <b>Route:</b> {idx + 1}<br>
            <b>Source:</b> Point {row.get('source', 'Unknown')}<br>
            <b>Target:</b> Point {row.get('target', 'Unknown')}<br>
            <b>Distance:</b> {row.get('distance', 0):.4f} degrees
            """
            
            # Convert distance to approximate km (at Calgary's latitude)
            dist_km = row.get('distance', 0) * 111  # 1 degree ≈ 111km at the equator
            popup_content += f"<br><b>Approx. Distance:</b> {dist_km:.2f} km"
            
            # Add the route line
            folium.PolyLine(
                locations=[(p[1], p[0]) for p in row.geometry.coords],  # (lat, lon) format
                color='blue',
                weight=4,
                opacity=0.7,
                tooltip=popup_content
            ).add_to(m)
    
    # Add high-demand points
    if demand_data is not None and 'future_demand' in demand_data.columns:
        # Get top demand points
        top_demand = demand_data.sort_values('future_demand', ascending=False).head(10)
        
        for idx, row in top_demand.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                tooltip=f"High Demand Point {idx}<br>Demand: {row.get('future_demand', 0):.2f}"
            ).add_to(m)
    
    # Add existing stations for reference
    if ev_stations is not None:
        for idx, row in ev_stations.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.7,
                tooltip=f"Existing Station: {row.get('station_name', 'Unknown')}"
            ).add_to(m)
    
    return m

def create_recommendations_map(recommendations, ev_stations=None):
    """
    Create a map showing recommended locations for new EV charging stations
    
    Parameters:
    -----------
    recommendations : DataFrame
        Ranked recommendations for station placement
    ev_stations : GeoDataFrame, optional
        Existing EV stations for reference
        
    Returns:
    --------
    folium.Map
        Folium map with recommendations
    """
    # Create a map centered on Calgary
    m = folium.Map(location=[51.05, -114.07], zoom_start=11)
    
    # Add recommended locations to the map
    if recommendations is not None:
        for idx, row in recommendations.iterrows():
            # Create popup content
            popup_content = f"""
            <b>Rank:</b> {idx + 1}<br>
            <b>Cluster:</b> {row.get('cluster_id', 'Unknown')}<br>
            <b>Score:</b> {row.get('combined_score', 0):.2f}<br>
            <b>Future Demand:</b> {row.get('future_demand', 0):.2f}<br>
            """
            
            if 'nearest_community' in recommendations.columns:
                popup_content += f"<b>Nearest Community:</b> {row.get('nearest_community', 'Unknown')}<br>"
                
            popup_content += f"<b>Data Source:</b> {row.get('source', 'Unknown')}"
            
            # Use gold markers for top 3, silver for 4-6, bronze for others
            if idx < 3:
                color = 'darkgoldenrod'
                radius = 12
            elif idx < 6:
                color = 'silver'
                radius = 10
            else:
                color = 'chocolate'
                radius = 8
            
            folium.CircleMarker(
                location=[row.get('latitude'), row.get('longitude')],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=popup_content
            ).add_to(m)
    
    # Add existing stations for reference
    if ev_stations is not None:
        for idx, row in ev_stations.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.7,
                tooltip=f"Existing Station: {row.get('station_name', 'Unknown')}"
            ).add_to(m)
    
    return m

def create_heatmap(data_points):
    """
    Create a heatmap of data points with extra robust error handling
    
    Parameters:
    -----------
    data_points : GeoDataFrame
        Data points with weight information
        
    Returns:
    --------
    folium.Map
        Folium map with heatmap
    """
    # Create a map centered on Calgary
    m = folium.Map(location=[51.05, -114.07], zoom_start=11)
    
    # Safety check for the input data
    if data_points is None or not isinstance(data_points, gpd.GeoDataFrame) or data_points.empty:
        folium.Marker(
            location=[51.05, -114.07],
            popup="No valid data for heatmap",
            icon=folium.Icon(color="orange", icon="info-sign")
        ).add_to(m)
        return m
    
    # Extract points with weights for the heatmap
    try:
        heatmap_data = []
        
        # First, check if the geometry column exists and contains Points
        if 'geometry' not in data_points.columns:
            folium.Marker(
                location=[51.05, -114.07],
                popup="No geometry column in data",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            return m
        
        # Process each row with extremely robust error handling
        for idx, row in data_points.iterrows():
            try:
                # Skip if geometry is missing or invalid
                if not hasattr(row, 'geometry') or row.geometry is None or row.geometry.is_empty:
                    continue
                
                # Ensure we have valid coordinates
                if not hasattr(row.geometry, 'y') or not hasattr(row.geometry, 'x'):
                    continue
                
                y, x = row.geometry.y, row.geometry.x
                
                # Skip if coordinates are NaN
                if pd.isna(y) or pd.isna(x):
                    continue
                
                # Ensure coordinates are valid numbers
                try:
                    y_float = float(y)
                    x_float = float(x)
                except (ValueError, TypeError):
                    continue
                
                # Base weight is 1.0
                weight = 1.0
                
                # Carefully get importance_score if available
                if 'importance_score' in data_points.columns:
                    try:
                        importance = row.get('importance_score', 1.0)
                        # Handle different data types safely
                        if isinstance(importance, (int, float)) and not pd.isna(importance):
                            weight = float(importance)
                        elif isinstance(importance, str):
                            # Try to convert string to float
                            try:
                                weight = float(importance)
                            except (ValueError, TypeError):
                                # Keep default if conversion fails
                                pass
                    except Exception:
                        # Ignore any error in processing importance
                        pass
                
                # Carefully get weight if available
                if 'weight' in data_points.columns:
                    try:
                        w = row.get('weight', 1.0)
                        # Handle different data types safely
                        if isinstance(w, (int, float)) and not pd.isna(w):
                            weight *= float(w)
                        elif isinstance(w, str):
                            # Try to convert string to float
                            try:
                                weight *= float(w)
                            except (ValueError, TypeError):
                                # Ignore if conversion fails
                                pass
                    except Exception:
                        # Ignore any error in processing weight
                        pass
                
                # Carefully get future_demand if available
                if 'future_demand' in data_points.columns:
                    try:
                        demand = row.get('future_demand', 0.0)
                        # Handle different data types safely
                        if isinstance(demand, (int, float)) and not pd.isna(demand):
                            weight *= (1.0 + float(demand))
                        elif isinstance(demand, str):
                            # Try to convert string to float
                            try:
                                weight *= (1.0 + float(demand))
                            except (ValueError, TypeError):
                                # Ignore if conversion fails
                                pass
                    except Exception:
                        # Ignore any error in processing demand
                        pass
                
                # Final safety check on weight - ensure it's a positive number
                if not isinstance(weight, (int, float)) or pd.isna(weight) or weight <= 0:
                    weight = 1.0
                
                # Add to heatmap data - ensure all values are float
                heatmap_data.append([float(y_float), float(x_float), float(weight)])
                
            except Exception as e:
                # Skip this point and continue with others - don't let one bad point ruin everything
                continue
        
        # Add the heatmap layer if we have data and it's not empty
        if len(heatmap_data) > 0:
            try:
                HeatMap(heatmap_data, 
                       min_opacity=0.2, 
                       radius=15, 
                       blur=10, 
                       gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
                      ).add_to(m)
            except Exception as e:
                # If heatmap creation fails, add a simpler visualization
                # Add a marker for each point instead
                for point in heatmap_data:
                    try:
                        folium.CircleMarker(
                            location=[point[0], point[1]],
                            radius=5,
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.7
                        ).add_to(m)
                    except:
                        # Skip this point if it can't be added
                        continue
                
                # Add a warning marker
                folium.Marker(
                    location=[51.05, -114.07],
                    popup=f"Fallback to simple markers due to heatmap error",
                    icon=folium.Icon(color="orange", icon="info-sign")
                ).add_to(m)
        else:
            # Add a marker indicating no valid data points
            folium.Marker(
                location=[51.05, -114.07],
                popup="No valid data points for heatmap",
                icon=folium.Icon(color="orange", icon="info-sign")
            ).add_to(m)
    
    except Exception as e:
        # Add a marker indicating error
        folium.Marker(
            location=[51.05, -114.07],
            popup=f"Error creating heatmap: {str(e)}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
    
    return m

def create_demand_growth_chart(growth_data):
    """
    Create a chart showing demand growth over time
    
    Parameters:
    -----------
    growth_data : dict
        Dictionary with growth metrics from calculate_demand_growth
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with demand growth chart
    """
    if not growth_data:
        # Return empty figure
        return go.Figure()
    
    # Create the chart
    fig = go.Figure()
    
    # Add the growth projection line
    fig.add_trace(go.Scatter(
        x=growth_data['months'],
        y=growth_data['demand_values'],
        mode='lines+markers',
        name='Projected Demand',
        line=dict(color='royalblue', width=3),
        marker=dict(size=8)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=growth_data['months'] + growth_data['months'][::-1],
        y=growth_data['upper_bound'] + growth_data['lower_bound'][::-1],
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval (±20%)"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"EV Charging Demand Growth Projection",
        xaxis_title="Forecast Period",
        yaxis_title="Relative Demand",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_cluster_chart(clustered_data):
    """
    Create a chart showing cluster size distribution
    
    Parameters:
    -----------
    clustered_data : GeoDataFrame
        Clustering results with cluster labels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with cluster distribution chart
    """
    if clustered_data is None or 'cluster' not in clustered_data.columns:
        # Return empty figure
        return go.Figure()
    
    # Get cluster counts
    cluster_counts = clustered_data[clustered_data['cluster'] >= 0]['cluster'].value_counts().sort_index()
    
    # Create a bar chart of cluster sizes
    cluster_df = pd.DataFrame({
        'Cluster': cluster_counts.index.astype(str),
        'Size': cluster_counts.values
    })
    
    fig = px.bar(
        cluster_df, x='Cluster', y='Size',
        title="Number of Points per Cluster",
        color='Size',
        color_continuous_scale='Viridis'
    )
    
    return fig

def create_route_distance_chart(route_stats):
    """
    Create a chart showing route distance distribution
    
    Parameters:
    -----------
    route_stats : dict
        Dictionary with route statistics from calculate_route_statistics
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with route distance chart
    """
    if not route_stats or 'distance_data' not in route_stats:
        # Return empty figure
        return go.Figure()
    
    # Create bar chart from distance data
    fig = px.bar(
        route_stats['distance_data'], 
        x='Route', 
        y='Distance (km)',
        title="Route Distances",
        color='Distance (km)',
        color_continuous_scale='Viridis'
    )
    
    return fig

def create_implementation_timeline():
    """
    Create a chart showing implementation timeline
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with implementation timeline
    """
    # Define phases for implementation
    timeline_data = {
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Monitoring'],
        'Start': [0, 3, 8, 12],
        'Duration': [3, 5, 4, 12]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.timeline(
        timeline_df, 
        x_start='Start',
        x_end=timeline_df['Start'] + timeline_df['Duration'],
        y='Phase',
        color='Phase',
        title="Implementation Timeline (Months)",
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(title="Month"),
        yaxis=dict(
            title="",
            categoryorder='array',
            categoryarray=timeline_df['Phase'][::-1]  # Reverse order
        ),
        height=300
    )
    
    return fig

def create_comparison_chart(current_stations_per_100k, recommended_stations, population):
    """
    Create a chart comparing Calgary's EV charging infrastructure with other cities
    
    Parameters:
    -----------
    current_stations_per_100k : float
        Current number of stations per 100,000 people
    recommended_stations : int
        Number of recommended new stations
    population : int
        Calgary's population
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with comparison chart
    """
    # Calculate proposed stations per 100k
    proposed_per_100k = current_stations_per_100k + (recommended_stations * 100000 / population)
    
    # Create comparison data
    comparison_data = {
        'City': ['Calgary (Current)', 'Calgary (Proposed)', 'Vancouver', 'Toronto', 'Montreal'],
        'Stations per 100k People': [current_stations_per_100k, proposed_per_100k, 15.2, 11.8, 9.5]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df,
        x='City',
        y='Stations per 100k People',
        title="EV Charging Infrastructure Comparison",
        color='City',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    )
    
    return fig