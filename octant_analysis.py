import pandas as pd
import geopandas as gpd
import numpy as np
# Remove matplotlib dependency
import folium
from folium.plugins import MarkerCluster
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon

def divide_into_octants(data_points, center_point=None):
    """
    Divide Calgary's geography into 8 octants and assign data points to these octants
    
    Parameters:
    -----------
    data_points : GeoDataFrame
        Spatial data points to analyze
    center_point : tuple, optional
        (lat, lon) of center point, defaults to Calgary downtown
        
    Returns:
    --------
    GeoDataFrame
        Data with octant assignments
    list
        List of octant polygons
    """
    if data_points is None or len(data_points) == 0:
        return None, []
    
    # Use Calgary downtown as center if not specified
    if center_point is None:
        center_point = (51.05, -114.07)  # Calgary downtown coordinates
    
    # Create a copy of the data to avoid modifying the original
    data_with_octants = data_points.copy()
    
    # Calculate octant for each point
    def assign_octant(point, center):
        # Skip points with invalid geometry
        if point is None or not hasattr(point, 'x') or not hasattr(point, 'y'):
            return -1
            
        # Calculate angle from center point
        dx = point.x - center[1]  # lon
        dy = point.y - center[0]  # lat
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize angle to 0-360
        angle = (angle + 360) % 360
        
        # Assign octant (0-7) based on angle
        # Octant 0: 0-45, Octant 1: 45-90, etc.
        return int(angle // 45)
    
    # Apply octant assignment to each point
    data_with_octants['octant'] = data_with_octants.geometry.apply(
        lambda p: assign_octant(p, center_point)
    )
    
    # Create octant polygons for visualization
    octant_polygons = []
    for i in range(8):
        # Calculate start and end angles for this octant
        start_angle = i * 45
        end_angle = (i + 1) * 45
        
        # Create polygon points
        polygon_points = [center_point]  # Start with center
        
        # Add points along the arc
        radius = 0.1  # degrees, approximately 10km
        for angle in np.linspace(start_angle, end_angle, 10):
            rad_angle = np.radians(angle)
            x = center_point[1] + radius * np.cos(rad_angle)
            y = center_point[0] + radius * np.sin(rad_angle)
            polygon_points.append((y, x))  # (lat, lon)
        
        # Close the polygon
        polygon_points.append(center_point)
        
        # Create a polygon
        octant_polygons.append(Polygon(polygon_points))
    
    return data_with_octants, octant_polygons

def analyze_octant_demand(data_with_octants):
    """
    Analyze EV demand by octant
    
    Parameters:
    -----------
    data_with_octants : GeoDataFrame
        Data with octant assignments
        
    Returns:
    --------
    DataFrame
        Summary statistics by octant
    """
    if data_with_octants is None or 'octant' not in data_with_octants.columns:
        return None
    
    # Filter to valid octants
    valid_data = data_with_octants[data_with_octants['octant'] >= 0]
    
    # Create summary by octant
    octant_summary = []
    
    for octant in range(8):
        octant_data = valid_data[valid_data['octant'] == octant]
        
        # Count points in this octant
        point_count = len(octant_data)
        
        # Calculate demand metrics if available
        if 'current_demand' in octant_data.columns and 'future_demand' in octant_data.columns:
            current_demand = octant_data['current_demand'].sum()
            future_demand = octant_data['future_demand'].sum()
            growth = ((future_demand / current_demand) - 1) * 100 if current_demand > 0 else 0
        else:
            current_demand = 0
            future_demand = 0
            growth = 0
            
        # Calculate EV station count if source column is available
        if 'source' in octant_data.columns:
            ev_stations = len(octant_data[octant_data['source'] == 'ev_station'])
        else:
            ev_stations = 0
            
        # Get octant direction name
        direction_names = [
            "East (0°-45°)", 
            "Northeast (45°-90°)", 
            "North (90°-135°)", 
            "Northwest (135°-180°)",
            "West (180°-225°)", 
            "Southwest (225°-270°)", 
            "South (270°-315°)", 
            "Southeast (315°-360°)"
        ]
        
        # Create summary row
        summary = {
            'octant': octant,
            'direction': direction_names[octant],
            'point_count': point_count,
            'ev_stations': ev_stations,
            'current_demand': current_demand,
            'future_demand': future_demand,
            'demand_growth': growth
        }
        
        octant_summary.append(summary)
    
    # Create DataFrame from summary
    summary_df = pd.DataFrame(octant_summary)
    
    return summary_df

def create_octant_map(data_with_octants, octant_polygons, ev_stations=None):
    """
    Create a map showing octants with demand overlay
    
    Parameters:
    -----------
    data_with_octants : GeoDataFrame
        Data with octant assignments
    octant_polygons : list
        List of octant polygons
    ev_stations : GeoDataFrame, optional
        Existing EV stations for reference
        
    Returns:
    --------
    folium.Map
        Folium map with octants and demand visualization
    """
    # Create a map centered on Calgary
    m = folium.Map(location=[51.05, -114.07], zoom_start=11)
    
    # Add octant polygons to the map
    for i, polygon in enumerate(octant_polygons):
        # Direction names
        direction_names = [
            "East (0°-45°)", 
            "Northeast (45°-90°)", 
            "North (90°-135°)", 
            "Northwest (135°-180°)",
            "West (180°-225°)", 
            "Southwest (225°-270°)", 
            "South (270°-315°)", 
            "Southeast (315°-360°)"
        ]
        
        # Generate color based on octant
        color = f"#{hash(str(i)) % 0xFFFFFF:06x}"
        
        # Create popup content
        popup_content = f"Octant {i}: {direction_names[i]}"
        
        # Add polygon to map
        folium.GeoJson(
            polygon,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.2
            },
            tooltip=popup_content
        ).add_to(m)
    
    # Add points with demand information if available
    if data_with_octants is not None and 'future_demand' in data_with_octants.columns:
        # Create cluster groups
        clusters = MarkerCluster().add_to(m)
        
        for idx, row in data_with_octants.iterrows():
            if row.geometry is None or not hasattr(row.geometry, 'x'):
                continue
                
            # Skip points with no demand
            if row.get('future_demand', 0) <= 0:
                continue
                
            # Color based on octant
            if row.get('octant', -1) >= 0:
                color = f"#{hash(str(row.get('octant'))) % 0xFFFFFF:06x}"
            else:
                color = 'gray'
            
            # Create popup content
            popup_content = f"""
            <b>Octant:</b> {row.get('octant', 'Unknown')}<br>
            <b>Current Demand:</b> {row.get('current_demand', 0):.2f}<br>
            <b>Future Demand:</b> {row.get('future_demand', 0):.2f}<br>
            <b>Source:</b> {row.get('source', 'Unknown')}
            """
            
            # Add marker to cluster
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=popup_content
            ).add_to(clusters)
    
    # Add existing charging stations for reference
    if ev_stations is not None:
        for idx, row in ev_stations.iterrows():
            if row.geometry is None or not hasattr(row.geometry, 'x'):
                continue
                
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

def create_octant_demand_chart(octant_summary):
    """
    Create a chart showing demand by octant
    
    Parameters:
    -----------
    octant_summary : DataFrame
        Summary statistics by octant
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with octant demand chart
    """
    if octant_summary is None:
        # Return empty figure
        return go.Figure()
    
    # Create radar chart for octant demand
    fig = go.Figure()
    
    # Add current demand trace
    fig.add_trace(go.Scatterpolar(
        r=octant_summary['current_demand'],
        theta=octant_summary['direction'],
        fill='toself',
        name='Current Demand',
        line_color='#1f77b4'
    ))
    
    # Add future demand trace
    fig.add_trace(go.Scatterpolar(
        r=octant_summary['future_demand'],
        theta=octant_summary['direction'],
        fill='toself',
        name='Future Demand',
        line_color='#ff7f0e'
    ))
    
    # Update layout
    fig.update_layout(
        title='EV Charging Demand by Road Network Octant',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(octant_summary['future_demand']) * 1.1]
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_octant_station_chart(octant_summary):
    """
    Create a chart showing existing stations by octant
    
    Parameters:
    -----------
    octant_summary : DataFrame
        Summary statistics by octant
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with octant stations chart
    """
    if octant_summary is None:
        # Return empty figure
        return go.Figure()
    
    # Create bar chart for existing stations by octant
    fig = px.bar(
        octant_summary,
        x='direction',
        y='ev_stations',
        color='demand_growth',
        title='EV Charging Stations by Road Network Octant',
        labels={'direction': 'Octant', 'ev_stations': 'Number of Stations', 'demand_growth': 'Demand Growth (%)'},
        color_continuous_scale='RdYlGn'
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(title='Number of Stations')
    )
    
    return fig

def create_octant_demand_gap_chart(octant_summary):
    """
    Create a chart showing the gap between demand and station coverage
    
    Parameters:
    -----------
    octant_summary : DataFrame
        Summary statistics by octant
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with octant demand gap chart
    """
    if octant_summary is None:
        # Return empty figure
        return go.Figure()
    
    # Calculate demand per station ratio
    octant_summary['demand_per_station'] = octant_summary.apply(
        lambda x: x['future_demand'] / max(x['ev_stations'], 1), axis=1
    )
    
    # Create horizontal bar chart
    fig = px.bar(
        octant_summary.sort_values('demand_per_station', ascending=False),
        y='direction',
        x='demand_per_station',
        color='demand_per_station',
        title='Demand-to-Station Ratio by Road Network Octant',
        labels={'direction': 'Octant', 'demand_per_station': 'Future Demand per Station'},
        orientation='h',
        color_continuous_scale='YlOrRd'
    )
    
    # Add a vertical line for average
    avg_demand_per_station = octant_summary['demand_per_station'].mean()
    
    fig.add_vline(
        x=avg_demand_per_station,
        line_dash="dash",
        line_color="green",
        annotation_text="Average",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(title='Future Demand per Station (higher = more need)'),
        yaxis=dict(title='')
    )
    
    return fig