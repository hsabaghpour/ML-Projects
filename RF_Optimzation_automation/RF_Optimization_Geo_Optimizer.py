import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime

# Load the data from a CSV file
data = pd.read_csv('network_kpi_data.csv')

# Ensure the date column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Define the threshold values for degraded KPIs
thresholds = {
    'call_drop_rate': 3.0,  # percentage
    'availability': 99.0,   # percentage
    'throughput': 10.0,     # Mbps
    'accessibility': 95.0   # percentage
}

# Identify the worst-performing cells based on the thresholds
worst_cells = data[
    (data['call_drop_rate'] > thresholds['call_drop_rate']) |
    (data['availability'] < thresholds['availability']) |
    (data['throughput'] < thresholds['throughput']) |
    (data['accessibility'] < thresholds['accessibility'])
]

# Extract the geographical coordinates of the worst-performing cells
coordinates = worst_cells[['latitude', 'longitude']].values

# Function to calculate the distance matrix
def calculate_distance_matrix(locations):
    distance_matrix = np.zeros((len(locations), len(locations)))
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            distance_matrix[i, j] = geodesic(loc1, loc2).kilometers
    return distance_matrix

# Calculate the distance matrix for the worst-performing cells
distance_matrix = calculate_distance_matrix(coordinates)

# Perform DBSCAN clustering with a maximum distance of 2 km
dbscan = DBSCAN(eps=2, min_samples=2, metric='precomputed')
labels = dbscan.fit_predict(distance_matrix)

# Add the cluster labels to the worst-performing cells DataFrame
worst_cells['cluster'] = labels

# Label the top offenders (worst-performing and geographically correlated cells)
worst_cells['top_offender'] = worst_cells['cluster'] != -1

# Merge the top offender labels back into the original data
data = data.merge(worst_cells[['cell_id', 'top_offender']], on='cell_id', how='left')
data['top_offender'] = data['top_offender'].fillna(False)

# Aggregation to daily, weekly, and monthly levels
data['date'] = pd.to_datetime(data['date'])
daily_data = data.groupby([data['date'].dt.date, 'cell_id']).mean().reset_index()
weekly_data = data.groupby([data['date'].dt.to_period('W').apply(lambda r: r.start_time), 'cell_id']).mean().reset_index()
monthly_data = data.groupby([data['date'].dt.to_period('M').apply(lambda r: r.start_time), 'cell_id']).mean().reset_index()

# Compare the data to see if there are any degradations compared to the last week
def compare_weeks(data):
    data['week'] = data['date'].dt.to_period('W').apply(lambda r: r.start_time)
    current_week = data['week'].max()
    previous_week = current_week - pd.Timedelta(weeks=1)

    current_week_data = data[data['week'] == current_week]
    previous_week_data = data[data['week'] == previous_week]

    comparison = current_week_data.set_index('cell_id').subtract(previous_week_data.set_index('cell_id'), fill_value=0)
    return comparison

weekly_comparison = compare_weeks(data)

# Plotting functions
def plot_kpis_over_time(data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    for kpi in ['call_drop_rate', 'availability', 'throughput', 'accessibility']:
        ax.plot(data['date'], data[kpi], label=kpi)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_kpis_over_time(daily_data, 'Daily KPI Trends')
plot_kpis_over_time(weekly_data, 'Weekly KPI Trends')
plot_kpis_over_time(monthly_data, 'Monthly KPI Trends')

# Traffic growth prediction and alert for capacity expansion
def predict_traffic_growth(data):
    data['timestamp'] = data['date'].apply(lambda x: x.timestamp())
    X = data[['timestamp']]
    y = data['throughput']

    model = LinearRegression()
    model.fit(X, y)
    
    future_dates = pd.date_range(start=data['date'].max(), periods=30, freq='D')
    future_timestamps = future_dates.map(lambda x: x.timestamp()).reshape(-1, 1)
    future_predictions = model.predict(future_timestamps)

    return future_dates, future_predictions

future_dates, future_predictions = predict_traffic_growth(data)

# Plot traffic growth predictions
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['throughput'], label='Historical Throughput')
plt.plot(future_dates, future_predictions, label='Predicted Throughput', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Throughput (Mbps)')
plt.title('Traffic Growth Prediction')
plt.legend()
plt.show()

# Alerting for capacity expansion
def check_capacity_expansion(predictions, threshold=15):
    if np.any(predictions > threshold):
        print("Alert: Predicted traffic growth exceeds capacity threshold! Consider expanding capacity.")
    else:
        print("No capacity expansion needed based on predicted traffic growth.")

check_capacity_expansion(future_predictions)

# Save the results to a new CSV file
data.to_csv('top_offenders_with_geographical_correlation.csv', index=False)
daily_data.to_csv('daily_data.csv', index=False)
weekly_data.to_csv('weekly_data.csv', index=False)
monthly_data.to_csv('monthly_data.csv', index=False)
weekly_comparison.to_csv('weekly_comparison.csv', index=False)

print("All analyses have been completed and results saved to CSV files.")
