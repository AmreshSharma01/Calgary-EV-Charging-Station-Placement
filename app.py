import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import warnings
import os
import plotly.express as px  # Add this import
import plotly.graph_objects as go 

# Import modules
from data_loader import load_all_datasets, prepare_combined_data
from spatial_analysis import perform_clustering, generate_recommendations
from demand_prediction import predict_demand, calculate_demand_growth
from route_optimization import optimize_routes, calculate_route_statistics
from visualization import (
    create_folium_map, create_cluster_map, create_demand_map, create_routes_map,
    create_recommendations_map, create_heatmap, create_demand_growth_chart,
    create_cluster_chart, create_route_distance_chart, create_implementation_timeline,
    create_comparison_chart
)
from utils import format_recommendations_table, calculate_stations_per_100k, get_feature_info

# Make sure to install matplotlib if you haven't already:
# pip install matplotlib
from octant_analysis import (
    divide_into_octants, analyze_octant_demand, create_octant_map,
    create_octant_demand_chart, create_octant_station_chart, create_octant_demand_gap_chart
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EV Charging Station Placement Optimizer - Calgary",
    page_icon="🔌",
    layout="wide"
)

# Title and introduction
st.title("🔌 Calgary EV Charging Station Placement Optimizer")
st.markdown("""
This application helps urban planners identify optimal locations for new EV charging 
stations using spatial data mining techniques. By analyzing traffic patterns, demographics, and existing 
infrastructure in Calgary, we provide data-driven recommendations for EV charging station placement.
""")

# Enhanced welcome message with instructions
with st.expander("📋 How to Use This Application", expanded=True):
    st.markdown("""
    ### Getting Started
    1. **Explore Data**: Navigate through the Data Explorer tab to understand the current infrastructure.
    2. **Configure Parameters**: Adjust clustering algorithm, forecast period, and optimization settings.
    3. **Run Analysis**: Click the "Run Analysis" button to generate results.
    4. **Explore Results**: Navigate through the tabs to view different aspects of the analysis.
    
    ### Analysis Methods
    - **Clustering Analysis**: Identifies high-potential areas for charging stations
    - **Demand Prediction**: Forecasts future EV charging demand
    - **Route Optimization**: Optimizes routes between high-demand locations
    - **Octant Analysis**: Analyzes demand by road network octants
    - **Recommendations**: Ranked list of suggested locations for new charging stations
    """)

# Show application version and data update info
col1, col2 = st.columns(2)
with col1:
    st.caption("Application Version: 1.0.0")
with col2:
    st.caption("Data: Calgary, AB (2023-2025)")

# Add a divider before the rest of the content
st.divider()

# Initialize session state for storing analysis results
if 'combined_data' not in st.session_state:
    st.session_state.combined_data = None
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'demand_data' not in st.session_state:
    st.session_state.demand_data = None
if 'routes_data' not in st.session_state:
    st.session_state.routes_data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'quiet_mode' not in st.session_state:
    st.session_state.quiet_mode = False

# Create data directory if it doesn't exist
data_dir = "data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load datasets
ev_stations, traffic_volumes, community_points, population_data, road_network = load_all_datasets(data_dir)

# Display warning if any data is missing
missing_data = []
if ev_stations is None:
    missing_data.append("EV Charging Stations")
if traffic_volumes is None:
    missing_data.append("Traffic Volumes")
if community_points is None:
    missing_data.append("Community Points")
if population_data is None:
    missing_data.append("Population Data")
if road_network is None:
    missing_data.append("Road Network")

if missing_data:
    st.warning(f"The following datasets are missing: {', '.join(missing_data)}. Please ensure all data files are in the 'data/' directory.")

# Create sidebar for parameters
st.sidebar.title("Analysis Parameters")

# Spatial Analysis Parameters
st.sidebar.subheader("Spatial Analysis")
clustering_method = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["DBSCAN", "K-Means"]
)

if clustering_method == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (km)", 0.1, 5.0, 1.0, 0.1)
    min_samples = st.sidebar.slider("Minimum Samples", 2, 20, 5)
else:  # K-Means
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 8)

# Predictive Modeling Parameters
st.sidebar.subheader("Predictive Modeling")
forecast_period = st.sidebar.slider("Forecast Period (months)", 1, 24, 12)

# Route Optimization
st.sidebar.subheader("Route Optimization")
route_algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["Dijkstra's Algorithm", "A* Search"]
)

# Run Analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Advanced settings
st.sidebar.subheader("Advanced Settings")
st.session_state.quiet_mode = st.sidebar.checkbox("Quiet Mode (Less Output)", value=False)

# Create tabs for different views
tabs = st.tabs(["Dashboard", "Data Explorer", "Clustering Analysis", "Demand Prediction", "Route Optimization", "Octant Analysis", "Recommendations"])

# Dashboard Tab
with tabs[0]:
    st.header("Dashboard")
    st.write("Welcome to the Calgary EV Charging Station Placement Dashboard. Get a quick overview of key insights.")
    
    # Create the main metrics layout
    st.subheader("Key Metrics")
    
    # Extract key metrics
    num_existing_stations = len(ev_stations) if ev_stations is not None else 0

    # Get population from data if available
    population = 1306784  # Default from 2021 census
    if population_data is not None:
        # Try to extract population from the data
        try:
            # Look for the population value in the population dataset
            # First try the specific format from the new dataset
            pop_rows = population_data[population_data['Characteristic'].str.contains('Population, 2021', case=False, na=False)]
            if len(pop_rows) > 0:
                pop_value = pop_rows.iloc[0]['Value']  # In new dataset format, the column is named 'Value'
                if isinstance(pop_value, (int, float)) or (isinstance(pop_value, str) and pop_value.replace(',', '').isdigit()):
                    # Remove commas if present (in case formatted as string with commas)
                    population = int(float(str(pop_value).replace(',', '')))
            
            # If we didn't find it with 'Value' column, try 'Total' column (old format)
            if population == 1306784 and 'Total' in population_data.columns:
                pop_rows = population_data[population_data['Characteristic'].str.contains('Population, 2021', case=False, na=False)]
                if len(pop_rows) > 0:
                    pop_value = pop_rows.iloc[0]['Total']
                    if isinstance(pop_value, (int, float)) or (isinstance(pop_value, str) and str(pop_value).replace(',', '').isdigit()):
                        population = int(float(str(pop_value).replace(',', '')))
        except Exception as e:
            if not st.session_state.get('quiet_mode', False):
                st.warning(f"Error extracting population: {str(e)}")

    # Calculate stations per 100,000 people
    stations_per_100k = calculate_stations_per_100k(num_existing_stations, population)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Existing Charging Stations", num_existing_stations)

    with col2:
        st.metric("Calgary Population", f"{population:,}")

    with col3:
        st.metric("Stations per 100k People", f"{stations_per_100k:.2f}")

    # Add income metrics section
    st.subheader("Household Income Metrics")
    income_cols = st.columns(2)

    # Extract income data if available
    avg_household_income = 129000  # Default from 2021 census
    median_household_income = 98000  # Default from 2021 census

    if population_data is not None:
        try:
            # Look for average income in the dataset
            avg_income_rows = population_data[population_data['Characteristic'].str.contains('Average total income of household', case=False, na=False)]
            if len(avg_income_rows) > 0:
                avg_value = avg_income_rows.iloc[0].get('Value', avg_income_rows.iloc[0].get('Total', 0))
                if isinstance(avg_value, (int, float)) or (isinstance(avg_value, str) and str(avg_value).replace(',', '').replace('$', '').strip().isdigit()):
                    avg_household_income = int(float(str(avg_value).replace(',', '').replace('$', '').strip()))
            
            # Look for median income in the dataset
            median_income_rows = population_data[population_data['Characteristic'].str.contains('Median total income of household', case=False, na=False)]
            if len(median_income_rows) > 0:
                median_value = median_income_rows.iloc[0].get('Value', median_income_rows.iloc[0].get('Total', 0))
                if isinstance(median_value, (int, float)) or (isinstance(median_value, str) and str(median_value).replace(',', '').replace('$', '').strip().isdigit()):
                    median_household_income = int(float(str(median_value).replace(',', '').replace('$', '').strip()))
        except Exception as e:
            if not st.session_state.get('quiet_mode', False):
                st.warning(f"Error extracting income data: {str(e)}")

    with income_cols[0]:
        st.metric("Average Household Income", f"${avg_household_income:,}")

    with income_cols[1]:
        st.metric("Median Household Income", f"${median_household_income:,}")

    # Add a note about income's impact on EV adoption
    with st.expander("🔍 Income & EV Adoption Correlation"):
        st.markdown("""
        ### Impact of Income on EV Adoption
        
        Household income is a strong predictor of EV adoption rates:
        
        - Higher-income households are 3-5× more likely to purchase EVs
        - Areas with median household income >$100,000 show 2× adoption rates
        - Income is factored into our demand prediction models
        
        This factor is weighted in our station placement recommendations to ensure balanced infrastructure development.
        """)

    # Create income distribution visualization if income data is available
    if population_data is not None:
        try:
            # Look for income distribution in the dataset
            income_dist_rows = population_data[population_data['Characteristic'].str.contains('Household total income groups in 2020', case=False, na=False)]
            if len(income_dist_rows) > 0:
                # Get all rows that have income ranges
                income_ranges = population_data[population_data['Characteristic'].str.contains('^\$', regex=True)]
                
                if len(income_ranges) > 0:
                    # Create dataframe for income distribution
                    income_dist_data = {
                        'Income Range': income_ranges['Characteristic'].tolist(),
                        'Households': income_ranges['Value'].astype(float).tolist() if 'Value' in income_ranges.columns else income_ranges['Total'].astype(float).tolist()
                    }
                    
                    # Create income distribution chart
                    st.subheader("Household Income Distribution")
                    
                    income_df = pd.DataFrame(income_dist_data)
                    fig = px.bar(
                        income_df,
                        x='Income Range',
                        y='Households',
                        title="Calgary Household Income Distribution (2020)",
                        color='Households',
                        color_continuous_scale='Viridis'
                    )
                    
                    # Improve layout
                    fig.update_layout(
                        xaxis=dict(tickangle=45),
                        yaxis=dict(title='Number of Households')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            if not st.session_state.get('quiet_mode', False):
                st.warning(f"Could not create income distribution chart: {str(e)}")

    
    # Create a map showing existing stations
    st.subheader("EV Charging Station Map")
    
    map_col1, map_col2 = st.columns(2)
    
    with map_col1:
        st.write("Existing Charging Stations")
        if ev_stations is not None:
            ev_map = create_folium_map(ev_stations, "Charging Stations", "green")
            folium_static(ev_map)
        else:
            st.info("No EV station data available.")
    
    with map_col2:
        st.write("Community Distribution")
        if community_points is not None:
            community_map = create_folium_map(community_points, "Communities", "blue")
            folium_static(community_map)
        else:
            st.info("No community data available.")
    
    # Add quick insights section
    st.subheader("Quick Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Create a chart showing distribution of stations by type
        st.write("Station Distribution by Type")
        
        if ev_stations is not None and 'station_type' in ev_stations.columns:
            station_type_counts = ev_stations['station_type'].value_counts().reset_index()
            station_type_counts.columns = ['Station Type', 'Count']
            
            fig = px.bar(
                station_type_counts,
                x='Station Type',
                y='Count',
                color='Station Type',
                title="Charging Station Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No station type data available.")
    
    with insights_col2:
        # Create a chart showing community distribution
        st.write("Community Distribution by Class")
        
        if community_points is not None and 'CLASS' in community_points.columns:
            community_class_counts = community_points['CLASS'].value_counts().reset_index()
            community_class_counts.columns = ['Class', 'Count']
            
            fig = px.pie(
                community_class_counts,
                values='Count',
                names='Class',
                title="Community Types",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No community class data available.")

# Data Explorer Tab
with tabs[1]:
    st.header("Data Explorer")
    
    # Create a row for selecting which dataset to view
    dataset_selection = st.selectbox(
        "Select dataset to view:",
        ["EV Charging Stations", "Traffic Volumes", "Community Points", "Population Data", "Road Network"]
    )
    
    # Show the selected dataset
    if dataset_selection == "EV Charging Stations":
        st.subheader("EV Charging Stations Data")
        if ev_stations is not None:
            st.dataframe(ev_stations.drop('geometry', axis=1, errors='ignore').head(10))
            st.write(f"Total stations: {len(ev_stations)}")
            
            # Map view of EV stations
            st.subheader("EV Stations Map")
            ev_map = create_folium_map(ev_stations, "Charging Stations", "green")
            folium_static(ev_map)
            
            # Show station distribution by type
            if 'station_type' in ev_stations.columns:
                st.subheader("Station Type Distribution")
                station_counts = ev_stations['station_type'].value_counts()
                fig = px.pie(values=station_counts.values, names=station_counts.index, 
                            title="Distribution of Charging Station Types")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No EV station data available.")
    
    elif dataset_selection == "Traffic Volumes":
        st.subheader("Traffic Volumes Data")
        if traffic_volumes is not None:
            st.dataframe(traffic_volumes.drop('geometry', axis=1, errors='ignore').head(10))
            st.write(f"Total records: {len(traffic_volumes)}")
            
            # Create a visual of traffic volumes
            if 'Volume' in traffic_volumes.columns:
                st.subheader("Traffic Volume Distribution")
                fig = px.histogram(traffic_volumes, x='Volume', nbins=20, 
                                   title="Distribution of Traffic Volumes")
                st.plotly_chart(fig, use_container_width=True)
                
                # Top traffic sections
                st.subheader("Top 10 Highest Traffic Volume Sections")
                top_traffic = traffic_volumes.sort_values('Volume', ascending=False).head(10)
                st.table(top_traffic[['Section Name', 'Volume']].reset_index(drop=True))
        else:
            st.info("No traffic volume data available.")
    
    elif dataset_selection == "Community Points":
        st.subheader("Community Points Data")
        if community_points is not None:
            st.dataframe(community_points.drop('geometry', axis=1, errors='ignore').head(10))
            st.write(f"Total communities: {len(community_points)}")
            
            # Map view of communities
            st.subheader("Communities Map")
            community_map = create_folium_map(community_points, "Communities", "blue")
            folium_static(community_map)
            
            # Distribution by class
            if 'CLASS' in community_points.columns:
                st.subheader("Community Class Distribution")
                class_counts = community_points['CLASS'].value_counts()
                fig = px.bar(x=class_counts.index, y=class_counts.values, 
                             labels={'x': 'Class', 'y': 'Count'},
                             title="Distribution of Community Classes")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No community data available.")
    
    elif dataset_selection == "Population Data":
        st.subheader("Population Data")
        if population_data is not None:
            st.dataframe(population_data.head(10))
            st.write(f"Total records: {len(population_data)}")
            
            # Try to find population growth information
            if 'Characteristic' in population_data.columns and 'Total' in population_data.columns:
                pop_rows = population_data[population_data['Characteristic'].str.contains('Population', case=False, na=False)]
                if len(pop_rows) > 0:
                    st.subheader("Population Information")
                    st.table(pop_rows[['Characteristic', 'Total']].reset_index(drop=True))
        else:
            st.info("No population data available.")
    
    elif dataset_selection == "Road Network":
        st.subheader("Road Network Data")
        if road_network is not None:
            st.dataframe(road_network.drop('geometry', axis=1, errors='ignore').head(10))
            st.write(f"Total road segments: {len(road_network)}")
            
            # Distribution by road class if available
            if 'CLASS_CODE' in road_network.columns:
                st.subheader("Road Class Distribution")
                road_class_counts = road_network['CLASS_CODE'].value_counts()
                fig = px.bar(x=road_class_counts.index, y=road_class_counts.values, 
                            labels={'x': 'Class Code', 'y': 'Count'},
                            title="Distribution of Road Classes")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No road network data available.")

# If run analysis button is clicked, show analysis results
if run_analysis:
    # Prepare combined data for analysis
    combined_data = prepare_combined_data(ev_stations, traffic_volumes, community_points)
    
    # Check if we have data to analyze
    if combined_data is None:
        st.error("No data available for analysis. Please check your datasets.")
    else:
        # Store combined data in session state for use across tabs
        st.session_state.combined_data = combined_data
        st.session_state.clustered_data = None
        st.session_state.demand_data = None
        st.session_state.routes_data = None
        st.session_state.recommendations = None
    
    # Clustering Analysis Tab
    with tabs[2]:
        st.header("Clustering Analysis")
        
        if combined_data is not None:
            try:
                with st.spinner("Performing spatial clustering..."):
                    # Perform clustering
                    if clustering_method == "DBSCAN":
                        st.subheader(f"DBSCAN Clustering (Epsilon: {eps} km, Min Samples: {min_samples})")
                        clustered_data = perform_clustering(
                            combined_data, method="DBSCAN", 
                            eps=eps, min_samples=min_samples
                        )
                    else:
                        st.subheader(f"K-Means Clustering (K: {n_clusters})")
                        clustered_data = perform_clustering(
                            combined_data, method="K-Means", 
                            n_clusters=n_clusters
                        )
                
# Update this section in the clustering analysis part of app.py
# This goes just after the clustering is performed

                if clustered_data is not None:
                    # Add debugging and data validation
                    try:
                        # Import the debug function from utils
                        from utils import debug_clustered_data
                        
                        # Debug and fix any datatype issues
                        fixed_data, debug_info = debug_clustered_data(clustered_data)
                        
                        # If there were issues, use the fixed data instead
                        if debug_info.get("fixed_columns") and len(debug_info["fixed_columns"]) > 0:
                            clustered_data = fixed_data
                            
                            if not st.session_state.get('quiet_mode', False):
                                st.info(f"Fixed datatype issues in the following columns: {', '.join(debug_info['fixed_columns'])}")
                    except Exception as e:
                        # If debugging fails, continue with original data
                        if not st.session_state.get('quiet_mode', False):
                            st.warning(f"Data validation encountered an error: {str(e)}")
                    
                    # Store for later use in other tabs
                    st.session_state.clustered_data = clustered_data
                    
                    # Display results
                    st.success(f"Clustering completed successfully!")
                    
                    # Create a map of the clustering results
                    st.subheader("Clustering Results Map")
                    
                    # Create cluster map
                    cluster_map = create_cluster_map(clustered_data, ev_stations)
                    folium_static(cluster_map)
                    
                    # Display cluster statistics
                    cluster_counts = clustered_data[clustered_data['cluster'] >= 0]['cluster'].value_counts().sort_index()
                    
                    st.subheader("Cluster Statistics")
                    
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        num_clusters = len(cluster_counts)
                        st.metric("Number of Clusters", num_clusters)
                    
                    with metric_cols[1]:
                        avg_points = int(cluster_counts.mean())
                        st.metric("Average Points per Cluster", avg_points)
                    
                    with metric_cols[2]:
                        noise_points = sum(clustered_data['cluster'] == -1) if 'cluster' in clustered_data.columns else 0
                        st.metric("Noise Points", noise_points)
                    
                    # Create a bar chart of cluster sizes
                    st.subheader("Cluster Size Distribution")
                    cluster_chart = create_cluster_chart(clustered_data)
                    st.plotly_chart(cluster_chart, use_container_width=True)
                    
                    # Create a heatmap of the data with additional fallback options
                    st.subheader("Spatial Density Heatmap")
                    try:
                        # First try the improved heatmap function
                        heatmap = create_heatmap(clustered_data)
                        folium_static(heatmap)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {str(e)}")
                        
                        # Provide a simpler alternative visualization if heatmap fails
                        st.info("Showing a simplified cluster density map instead.")
                        
                        try:
                            # Create a simpler point-based map as fallback
                            m = folium.Map(location=[51.05, -114.07], zoom_start=11)
                            
                            if clustered_data is not None and len(clustered_data) > 0:
                                for idx, row in clustered_data.iterrows():
                                    try:
                                        # Only process rows with valid geometry
                                        if row.geometry is None or not hasattr(row.geometry, 'y') or not hasattr(row.geometry, 'x'):
                                            continue
                                        
                                        # Skip if coordinates are NaN
                                        if pd.isna(row.geometry.y) or pd.isna(row.geometry.x):
                                            continue
                                        
                                        # Use a different color for each cluster
                                        if 'cluster' in row and row['cluster'] >= 0:
                                            # Generate a color based on cluster
                                            color = f"#{abs(hash(str(row['cluster']))) % 0xFFFFFF:06x}"
                                        else:
                                            color = 'gray'  # Default for noise points
                                        
                                        # Add circle marker with minimal properties
                                        folium.CircleMarker(
                                            location=[row.geometry.y, row.geometry.x],
                                            radius=5,
                                            color=color,
                                            fill=True,
                                            fill_color=color,
                                            fill_opacity=0.7,
                                            tooltip=f"Cluster: {row.get('cluster', 'None')}"
                                        ).add_to(m)
                                    except Exception:
                                        # Skip problematic points
                                        continue
                            
                            folium_static(m)
                            
                        except Exception as e2:
                            # If even the fallback fails, show a simple error message
                            st.error(f"Could not visualize cluster data: {str(e2)}")
                            st.info("Adjust the clustering parameters or try with different data.")                    
                else:
                    st.error("Clustering analysis failed. Please check your data.")
            
            except Exception as e:
                st.error(f"Error during clustering analysis: {str(e)}")
                st.info("Try adjusting the clustering parameters or check the data quality.")
        else:
            st.warning("No data available for clustering analysis. Please check your datasets.")
            
    # Demand Prediction Tab
    with tabs[3]:
        st.header("Demand Prediction")
        
        if combined_data is not None:
            try:
                with st.spinner("Analyzing demand patterns..."):
                    # Get features for prediction
                    features = get_feature_info()
                    
                    # Perform demand prediction
                    demand_data, accuracy = predict_demand(combined_data, features, forecast_period, population_data)
                
                if demand_data is not None:
                    # Store for later use
                    st.session_state.demand_data = demand_data
                    
                    # Display results
                    st.success(f"Demand prediction completed with {accuracy:.2f}% accuracy!")
                    
                    # Create map of predicted demand
                    st.subheader("Demand Prediction Map")
                    demand_map = create_demand_map(demand_data, ev_stations)
                    folium_static(demand_map)
                    
                    # Calculate growth metrics
                    growth_metrics = calculate_demand_growth(demand_data, forecast_period)
                    
                    # Show demand metrics
                    if growth_metrics:
                        st.subheader("Demand Forecast Metrics")
                        
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("Average Current Demand", f"{growth_metrics['avg_current']:.2f}")
                        
                        with metric_cols[1]:
                            st.metric("Average Future Demand", f"{growth_metrics['avg_future']:.2f}")
                            
                        with metric_cols[2]:
                            st.metric("Projected Growth", f"{growth_metrics['growth_percent']:.1f}%")
                        
                        # Create a chart comparing current vs future demand
                        st.subheader("Demand Growth Projection")
                        growth_chart = create_demand_growth_chart(growth_metrics)
                        st.plotly_chart(growth_chart, use_container_width=True)
                        
                        # Add contextual information about the forecast
                        with st.expander("About this Forecast"):
                            st.markdown(f"""
                            This forecast projects EV charging demand over the next {forecast_period} months based on:
                            
                            - **Current Charging Station Usage**: Utilization patterns of existing stations
                            - **Traffic Patterns**: Higher traffic areas tend to have higher charging demand
                            - **Community Factors**: Residential areas, especially newer ones, have higher EV adoption rates
                            - **Growth Rate**: We project a compound monthly growth rate of {growth_metrics['growth_rate']*100:.2f}%
                            
                            The confidence interval represents the uncertainty in these projections.
                            """)
                        with st.expander("Income's Influence on EV Adoption"):
                            st.markdown(f"""
                            ### How Income Affects Our Predictions
                            
                            Household income is incorporated in our model with the following assumptions:
                            
                            - Calgary's average household income of ${avg_household_income:,} contributes an income factor of {demand_data['income_factor'].mean():.2f} to our model
                            - Areas with higher income residents have faster EV adoption rates
                            - Income influences both current demand (proximity to higher-income areas) and growth rate projections
                            - Research shows income to be one of the strongest predictors of EV adoption, alongside age and education level
                            
                            Our model adjusts growth rates by up to ±10% based on the income factor, with higher-income areas showing accelerated adoption curves.
                            """)
                else:
                    st.error("Demand prediction failed. Please check your data.")
            
            except Exception as e:
                st.error(f"Error during demand prediction: {str(e)}")
                st.info("Try adjusting the forecast period or check the data quality.")
        else:
            st.warning("No data available for demand prediction. Please check your datasets.")
    
    # Route Optimization Tab
    with tabs[4]:
        st.header("Route Optimization")
        
        if hasattr(st.session_state, 'demand_data') and st.session_state.demand_data is not None:
            try:
                with st.spinner("Optimizing routes..."):
                    # Get high-demand points for routing
                    demand_data = st.session_state.demand_data
                    
                    # Perform route optimization
                    routes_data = optimize_routes(demand_data, road_network, algorithm=route_algorithm)
                
                if routes_data is not None and len(routes_data) > 0:
                    # Store for later use
                    st.session_state.routes_data = routes_data
                    
                    # Display results
                    st.success(f"Route optimization completed successfully!")
                    
                    # Create map of optimized routes
                    st.subheader("Optimized Routes Map")
                    routes_map = create_routes_map(routes_data, demand_data, ev_stations)
                    folium_static(routes_map)
                    
                    # Calculate route statistics
                    route_stats = calculate_route_statistics(routes_data)
                    
                    if route_stats:
                        # Display route statistics
                        st.subheader("Route Statistics")
                        
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("Number of Routes", route_stats['num_routes'])
                        
                        with metric_cols[1]:
                            st.metric("Average Route Distance", f"{route_stats['avg_distance']:.2f} km")
                        
                        with metric_cols[2]:
                            st.metric("Total Network Distance", f"{route_stats['total_distance']:.2f} km")
                        
                        # Create a distance distribution chart
                        st.subheader("Route Distance Distribution")
                        distance_chart = create_route_distance_chart(route_stats)
                        st.plotly_chart(distance_chart, use_container_width=True)
                        
                        # Add contextual information about route optimization
                        with st.expander("About Route Optimization"):
                            st.markdown(f"""
                            The route optimization identifies efficient connections between high-demand points:
                            
                            - **Algorithm Used**: {route_algorithm}
                            - **Optimization Goal**: Minimize travel distance between potential charging locations
                            - **Points Considered**: Top 10 high-demand locations from prediction model
                            - **Network Size**: {route_stats['num_routes']} routes with total length of {route_stats['total_distance']:.2f} km
                            
                            These routes help identify corridors where charging infrastructure would be most beneficial.
                            """)
                    
                else:
                    st.warning("Route optimization produced no results. Try adjusting parameters or using different data.")
            
            except Exception as e:
                st.error(f"Error during route optimization: {str(e)}")
                st.info("Try using a different algorithm or check if demand prediction was completed successfully.")
        else:
            st.warning("Demand prediction must be completed before route optimization. Run the analysis first.")
    
    # Octant Analysis Tab
    with tabs[5]:
        st.header("Road Network Octant Analysis")
        
        if hasattr(st.session_state, 'demand_data') and st.session_state.demand_data is not None:
            try:
                with st.spinner("Analyzing demand by road network octants..."):
                    # Get demand data
                    demand_data = st.session_state.demand_data
                    
                    # Divide into octants
                    data_with_octants, octant_polygons = divide_into_octants(demand_data)
                    
                    # Analyze demand by octant
                    octant_summary = analyze_octant_demand(data_with_octants)
                
                if octant_summary is not None:
                    # Display results
                    st.success(f"Octant analysis completed successfully!")
                    
                    # Create map with octants
                    st.subheader("Road Network Octant Map")
                    octant_map = create_octant_map(data_with_octants, octant_polygons, ev_stations)
                    folium_static(octant_map)
                    
                    # Display octant summary statistics
                    st.subheader("Octant Analysis Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Current EV stations by octant
                        st.metric("Total Octants Analyzed", "8")
                    
                    with col2:
                        # Calculate coverage metric
                        covered_octants = sum(octant_summary['ev_stations'] > 0)
                        st.metric("Octants with EV Stations", f"{covered_octants}/8")
                    
                    # Show the octant summary table
                    st.subheader("Demand and Station Distribution by Octant")
                    st.dataframe(
                        octant_summary[['direction', 'ev_stations', 'current_demand', 'future_demand', 'demand_growth']].round(2)
                    )
                    
                    # Create visualization charts
                    st.subheader("Visualizations")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Radar chart for demand by octant
                        demand_chart = create_octant_demand_chart(octant_summary)
                        st.plotly_chart(demand_chart, use_container_width=True)
                    
                    with chart_col2:
                        # Bar chart for stations by octant
                        station_chart = create_octant_station_chart(octant_summary)
                        st.plotly_chart(station_chart, use_container_width=True)
                    
                    # Full-width chart for demand gap
                    gap_chart = create_octant_demand_gap_chart(octant_summary)
                    st.plotly_chart(gap_chart, use_container_width=True)
                    
                    # Key insights and recommendations
                    st.subheader("Key Insights from Octant Analysis")
                    
                    # Find octants with highest demand and lowest coverage
                    high_demand_octants = octant_summary.sort_values('future_demand', ascending=False)['direction'].iloc[:2].tolist()
                    underserved_octants = octant_summary.sort_values('demand_per_station', ascending=False)['direction'].iloc[:2].tolist()
                    
                    st.markdown(f"""
                    ### Demand Distribution
                    - Highest demand areas are in the **{high_demand_octants[0]}** and **{high_demand_octants[1]}** octants
                    - Most underserved areas (highest demand-to-station ratio) are in the **{underserved_octants[0]}** and **{underserved_octants[1]}** octants
                    
                    ### Recommendations Based on Octant Analysis
                    1. **Prioritize new stations** in underserved octants with high demand-to-station ratios
                    2. **Balance the network** by ensuring at least minimal coverage in all octants
                    3. **Monitor growth patterns** in octants showing rapid demand increase
                    """)
                    
                    # Add contextual information about octant analysis
                    with st.expander("About Octant Analysis Methodology"):
                        st.markdown("""
                        The octant analysis divides Calgary's geography into 8 directional segments:
                        
                        1. **East (0°-45°)**
                        2. **Northeast (45°-90°)**
                        3. **North (90°-135°)**
                        4. **Northwest (135°-180°)**
                        5. **West (180°-225°)**
                        6. **Southwest (225°-270°)**
                        7. **South (270°-315°)**
                        8. **Southeast (315°-360°)**
                        
                        Each data point is assigned to an octant based on its position relative to Calgary's downtown center. This analysis helps identify:
                        
                        - **Geographic disparities** in charging infrastructure
                        - **Directional demand patterns** along major road network corridors
                        - **Underserved areas** where new stations would have the highest impact
                        - **Growth directions** where future expansion should be focused
                        
                        The demand-to-station ratio is a key metric that indicates which areas most urgently need additional charging infrastructure.
                        """)
                
            except Exception as e:
                st.error(f"Error during octant analysis: {str(e)}")
                st.info("Try running the demand prediction first or check for data issues.")
        else:
            st.warning("Demand prediction must be completed before octant analysis. Run the analysis first.")
    
    # Recommendations Tab
    with tabs[6]:
        st.header("Recommendations for EV Charging Station Placement")
        
        if (hasattr(st.session_state, 'clustered_data') and st.session_state.clustered_data is not None and 
            hasattr(st.session_state, 'demand_data') and st.session_state.demand_data is not None):
            
            try:
                with st.spinner("Generating recommendations..."):
# Generate recommendations based on clustering and demand data
                    recommendations = generate_recommendations(
                        st.session_state.clustered_data,
                        st.session_state.demand_data,
                        community_points
                    )
                    
                    # Store recommendations
                    st.session_state.recommendations = recommendations
                
                if recommendations is not None and len(recommendations) > 0:
                    # Display results
                    st.success(f"Generated {len(recommendations)} recommendations for new charging stations!")
                    
                    # Create a map with recommended locations
                    st.subheader("Recommended EV Charging Station Locations")
                    recommendations_map = create_recommendations_map(recommendations, ev_stations)
                    folium_static(recommendations_map)
                    
                    # Create a table with top recommendations
                    st.subheader("Top 10 Recommended Locations")
                    display_df = format_recommendations_table(recommendations.head(10))
                    st.table(display_df)
                    
                    # Add implementation plan
                    st.subheader("Implementation Plan")
                    
                    st.markdown("""
                    Based on our analysis, we recommend a phased implementation approach:
                    
                    ### Phase 1 (1-3 months)
                    - Install charging stations at the top 3 recommended locations
                    - Focus on high-demand, high-visibility areas
                    - Monitor usage patterns to validate model predictions
                    
                    ### Phase 2 (4-8 months)
                    - Expand to the next 4-7 recommended locations
                    - Adjust placement based on Phase 1 performance data
                    - Target different community types to ensure coverage
                    
                    ### Phase 3 (9-12 months)
                    - Complete network with remaining recommended locations
                    - Integrate with city-wide transportation planning
                    - Develop predictive maintenance and management system
                    """)
                    
                    # Create a timeline chart
                    timeline_chart = create_implementation_timeline()
                    st.plotly_chart(timeline_chart, use_container_width=True)
                    
                    # Add comparative analysis with similar cities
                    st.subheader("Comparative Analysis")
                    
                    # Create a comparison chart
                    comparison_chart = create_comparison_chart(
                        stations_per_100k,
                        len(recommendations),
                        population
                    )
                    st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # Add justification for recommendations
                    with st.expander("Recommendation Methodology"):
                        st.markdown("""
                        Our recommendations are based on a multi-factor analysis that considers:
                        
                        1. **Spatial Clustering**: Identifying natural groupings of potential high-demand areas
                        2. **Demand Forecasting**: Projecting future EV charging needs based on current patterns
                        3. **Route Optimization**: Ensuring strategic placement along common travel corridors
                        4. **Octant Analysis**: Balancing coverage across the eight directional segments of the city
                        5. **Existing Infrastructure**: Complementing current charging network
                        6. **Community Factors**: Considering demographics and community types
                        
                        Each recommended location is scored based on a combination of these factors, with higher weights given to future demand and strategic positioning.
                        """)
                    
                    # Final call to action
                    st.markdown("""
                    ### Next Steps
                    
                    1. **Validate Recommendations**: Conduct site visits to verify suitability
                    2. **Stakeholder Engagement**: Present findings to city planners and utility providers
                    3. **Funding Assessment**: Evaluate infrastructure costs and potential funding sources
                    4. **Implementation Planning**: Develop detailed rollout strategy with timelines
                    
                    By following these recommendations, Calgary can develop a robust, data-driven EV charging network that supports the growing adoption of electric vehicles.
                    """)
                    
                else:
                    st.warning("Could not generate recommendations from the analysis results.")
            
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.info("Ensure that clustering and demand prediction analyses completed successfully.")
        else:
            st.warning("Both clustering and demand prediction must be completed to generate recommendations. Run the analysis first.")
else:
    # If analysis hasn't been run, show message in other tabs
    for tab_idx, tab_name in enumerate(["Clustering Analysis", "Demand Prediction", "Route Optimization", "Octant Analysis", "Recommendations"]):
        if tab_idx + 2 < len(tabs):  # +2 because we're starting at the 3rd tab (index 2)
            with tabs[tab_idx + 2]:
                st.info(f"Click the 'Run Analysis' button in the sidebar to perform {tab_name.lower()}.")

# Add footer
st.divider()
st.caption("EV Charging Station Placement Optimizer - Developed for ENGO645 Project (Winter 2025)")
st.caption("Data sources: City of Calgary Open Data Portal")

# Run the app if this is the main file
if __name__ == "__main__":
    # The app is already running through Streamlit's execution model
    pass