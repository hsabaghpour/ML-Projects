GeoKPIAnalyzer: Uncover Your Network's Top Offenders! 📡
Welcome to GeoKPIAnalyzer! This awesome Python script is your new best friend for automating the discovery of the worst-performing cells in your LTE and 5G network. By combining KPI analysis with geographical clustering, GeoKPIAnalyzer identifies the top offenders and shows you exactly where they are. Let's dive in and see what it can do!

🌟 Features
KPI Analysis: Automatically pinpoints the worst-performing cells based on call drop rate, availability, throughput, and accessibility.
Geographical Clustering: Groups together the worst-performing cells that are within a 2 km radius using cutting-edge clustering algorithms.
Beautiful Visualization: Creates stunning scatter plots to help you visualize the geographically correlated top offenders.
Aggregation: Aggregates data at daily, weekly, and monthly levels.
Degradation Comparison: Compares weekly data to detect KPI degradations.
Traffic Growth Prediction: Predicts future traffic growth and alerts for capacity expansion needs.
🚀 Getting Started
Ready to get started? Awesome! Here's what you need to do:

Prerequisites
Make sure you have Python 3.6 or higher installed. You'll also need a few Python libraries. No worries, they're easy to get:

bash
Copy code
pip install pandas numpy geopy scikit-learn matplotlib
Data Preparation
Your data should be in a CSV file with these columns:

date (format: YYYY-MM-DD)
cell_id
call_drop_rate
availability
throughput
accessibility
latitude
longitude
Here's a quick example of what your CSV might look like:

plaintext
Copy code
date,cell_id,call_drop_rate,availability,throughput,accessibility,latitude,longitude
2023-01-01,cell_001,2.5,99.5,12.0,96.5,49.2827,-123.1207
2023-01-01,cell_002,3.5,98.0,9.0,94.0,49.2820,-123.1210
2023-01-01,cell_003,1.0,99.8,11.0,97.0,49.2825,-123.1205
2023-01-01,cell_004,4.0,97.5,8.5,93.0,49.2829,-123.1208
Running the Script
Ready to unleash GeoKPIAnalyzer? Run the script like this:

bash
Copy code
python geokpianalyzer.py
Results
Once the script finishes, you'll have several new CSV files with aggregated data and analysis results:

top_offenders_with_geographical_correlation.csv
daily_data.csv
weekly_data.csv
monthly_data.csv
weekly_comparison.csv
Plus, you'll get cool plots showing the KPI trends and traffic growth predictions.
