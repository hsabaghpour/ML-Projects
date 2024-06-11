GeoKPIAnalyzer: Uncover Your Network's Top Offenders! 📡
Welcome to GeoKPIAnalyzer! This awesome Python script is your new best friend for automating the discovery of the worst-performing cells in your LTE and 5G network. By combining KPI analysis with geographical clustering, GeoKPIAnalyzer identifies the top offenders and shows you exactly where they are. Let's dive in and see what it can do!

🌟 Features
KPI Analysis: Automatically pinpoints the worst-performing cells based on call drop rate, availability, throughput, and accessibility.
Geographical Clustering: Groups together the worst-performing cells that are within a 2 km radius using cutting-edge clustering algorithms.
Beautiful Visualization: Creates stunning scatter plots to help you visualize the geographically correlated top offenders.
🚀 Getting Started
Ready to get started? Awesome! Here's what you need to do:

Prerequisites
Make sure you have Python 3.6 or higher installed. You'll also need a few Python libraries. No worries, they're easy to get:

bash
Copy code
pip install pandas numpy geopy scikit-learn matplotlib
Data Preparation
Your data should be in a CSV file with these columns:

cell_id
call_drop_rate
availability
throughput
accessibility
latitude
longitude
cell_id,call_drop_rate,availability,throughput,accessibility,latitude,longitude
cell_001,2.5,99.5,12.0,96.5,49.2827,-123.1207
cell_002,3.5,98.0,9.0,94.0,49.2820,-123.1210
cell_003,1.0,99.8,11.0,97.0,49.2825,-123.1205
cell_004,4.0,97.5,8.5,93.0,49.2829,-123.1208
Running the Script
Ready to unleash GeoKPIAnalyzer? Run the script like this:

bash
Copy code
python geokpianalyzer.py
Results
Once the script finishes, you'll have a shiny new CSV file called top_offenders_with_geographical_correlation.csv. Plus, you'll get a cool scatter plot showing the top offenders.
