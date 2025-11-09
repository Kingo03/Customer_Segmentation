🎯 Customer Segmentation Predictor

An interactive Streamlit application that predicts customer segments using a pre-trained K-Means clustering model and provides rich data visualizations to explore customer behavior patterns.

This tool helps marketing teams and analysts understand different customer groups based on demographic, purchase, and behavioral data.

🚀 Features

✅ Interactive Data Input

Enter customer details across multiple tabs (Demographics, Purchases, Campaigns, etc.)

Predict the customer’s cluster instantly

✅ Cluster Prediction

Uses a trained scaler.pkl and kmeans.pkl model

Displays detailed cluster insights and visualizations

✅ Visualization Dashboard

Upload a full dataset to visualize segment distribution

Pie chart: Cluster distribution

PCA 2D plot: Visual cluster separation

Radar chart: Average feature profile per cluster

Heatmap: Feature differences across clusters

✅ Downloadable Prediction

Download individual predictions as CSV

🧠 Model Information

Algorithm: K-Means Clustering

Preprocessing: StandardScaler (stored in scaler.pkl)

Model File: kmeans.pkl

Optional Summary File: cluster_summary.csv (for displaying cluster statistics)
