import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the data from a CSV file
data = pd.read_csv('network_kpi_data.csv', parse_dates=['date'])

# Define the threshold values for degraded KPIs
thresholds = {
    'call_drop_rate': 3.0,  # percentage
    'availability': 99.0,   # percentage
    'throughput': 10.0,     # Mbps
    'accessibility': 95.0   # percentage
}

# Aggregation functions
def aggregate_data(df, freq):
    return df.resample(freq, on='date').mean()

# Detect degradation compared to last period
def detect_degradation(current, previous):
    return current > previous

# Predict traffic growth using Holt-Winters Exponential Smoothing
def predict_traffic(data, steps):
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(steps)
    return forecast

# Plotting functions
def plot_kpis(data, title):
    data.plot(figsize=(12, 6))
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.show()

# Identify worst-performing cells
def identify_worst_cells(data):
    return data[
        (data['call_drop_rate'] > thresholds['call_drop_rate']) |
        (data['availability'] < thresholds['availability']) |
        (data['throughput'] < thresholds['throughput']) |
        (data['accessibility'] < thresholds['accessibility'])
    ]

# Function to calculate the distance matrix
def calculate_distance_matrix(locations):
    distance_matrix = np.zeros((len(locations), len(locations)))
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            distance_matrix[i, j] = geodesic(loc1, loc2).kilometers
    return distance_matrix

# Perform geographical clustering
def geographical_clustering(coordinates):
    distance_matrix = calculate_distance_matrix(coordinates)
    dbscan = DBSCAN(eps=2, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)
    return labels

# Aggregate data to daily, weekly, and monthly levels
daily_data = aggregate_data(data, 'D')
weekly_data = aggregate_data(data, 'W')
monthly_data = aggregate_data(data, 'M')

# Detect degradation
last_week = weekly_data.iloc[-2]
current_week = weekly_data.iloc[-1]
degradation = detect_degradation(current_week, last_week)

# Predict traffic growth for the next month
traffic_forecast = predict_traffic(weekly_data['throughput'], steps=4)

# Identify worst-performing cells
worst_cells = identify_worst_cells(data)

# Extract coordinates of worst-performing cells
coordinates = worst_cells[['latitude', 'longitude']].values

# Perform geographical clustering
labels = geographical_clustering(coordinates)

# Add cluster labels to worst-performing cells
worst_cells['cluster'] = labels

# Label top offenders
worst_cells['top_offender'] = worst_cells['cluster'] != -1

# Merge top offender labels back into the original data
data = data.merge(worst_cells[['cell_id', 'top_offender']], on='cell_id', how='left')
data['top_offender'] = data['top_offender'].fillna(False)

# Save results to CSV
data.to_csv('top_offenders_with_geographical_correlation.csv', index=False)

# Plot aggregated data and forecast
plot_kpis(daily_data[['call_drop_rate', 'availability', 'throughput', 'accessibility']], 'Daily KPI Trends')
plot_kpis(weekly_data[['call_drop_rate', 'availability', 'throughput', 'accessibility']], 'Weekly KPI Trends')
plot_kpis(monthly_data[['call_drop_rate', 'availability', 'throughput', 'accessibility']], 'Monthly KPI Trends')
plot_kpis(traffic_forecast, 'Predicted Traffic Growth for Next Month')

# Plot geographical clusters of top offenders
top_offenders = data[data['top_offender']]
plt.figure(figsize=(10, 6))
plt.scatter(data['longitude'], data['latitude'], c='grey', alpha=0.5, label='All Sites')
plt.scatter(top_offenders['longitude'], top_offenders['latitude'], c='red', label='Top Offenders')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographically Correlated Top Offenders')
plt.legend()
plt.show()

print("Analysis complete! Results saved to 'top_offenders_with_geographical_correlation.csv'.")
