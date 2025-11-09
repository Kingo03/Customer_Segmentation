import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

# -----------------------------
# Load trained models
# -----------------------------
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

try:
    cluster_summary = pd.read_csv('cluster_summary.csv')
except FileNotFoundError:
    cluster_summary = pd.DataFrame()

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="🎯 Customer Segmentation Predictor", layout="wide")
st.title("🎯 Customer Segmentation Predictor")
st.markdown("Use this app to **predict customer segments** and explore insights from your clustering model.")

st.sidebar.header("🔧 About the Model")
st.sidebar.info("""
This app uses **K-Means clustering** with pre-trained scaling.
Upload customer details, and it predicts which cluster they belong to.
""")

# -----------------------------
# Feature groups
# -----------------------------
id_feature = ['ID']
demographics = ['Year_Birth', 'Education_nums', 'Marital_Status_nums', 'Income']
children_info = ['Kidhome', 'Teenhome', 'TotalChildren', 'HasChildren']
purchases = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
             'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
             'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
behaviour = ['Recency', 'NumWebVisitsMonth', 'Complain', 'Days_Since', 'RecencyRatio', 'TotalAccepted', 'Age']

all_features = id_feature + demographics + children_info + purchases + campaigns + behaviour

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🧾 Enter Customer Details")

tabs = st.tabs(["👤 Demographics", "👶 Family Info", "🛒 Purchases", "📢 Campaigns", "⚙️ Behavior"])

user_input = {}

user_input['ID'] = st.number_input("Customer ID", min_value=0, value=0)

with tabs[0]:
    st.write("### Demographic Information")
    for col in demographics:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

with tabs[1]:
    st.write("### Family / Children Info")
    for col in children_info:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

with tabs[2]:
    st.write("### Purchase History")
    for col in purchases:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

with tabs[3]:
    st.write("### Campaign Response")
    for col in campaigns:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

with tabs[4]:
    st.write("### Behavioral Info")
    for col in behaviour:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("---")
if st.button("🚀 Predict Customer Cluster"):
    with st.spinner("Analyzing customer profile..."):
        input_df = pd.DataFrame([user_input])

        # Align features with scaler training order
        input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

        scaled_input = scaler.transform(input_df)
        cluster = kmeans.predict(scaled_input)[0]
    
    st.success(f"🎉 Predicted Customer Cluster: **{cluster}**")

    # Cluster insights
    if not cluster_summary.empty:
        st.write(f"### 📊 Cluster {cluster} Overview")
        cluster_info = cluster_summary[cluster_summary['Cluster'] == cluster]

        if not cluster_info.empty:
            view_mode = st.radio("Choose visualization type:", ["📈 Bar Chart", "🕸️ Radar Chart"], horizontal=True)
            
            features = cluster_info.columns[1:]  # Skip 'Cluster' column
            values = cluster_info.iloc[0, 1:].values

            if view_mode == "📈 Bar Chart":
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(features, values, color='teal', alpha=0.7)
                ax.set_xticklabels(features, rotation=45, ha='right')
                ax.set_title(f"Cluster {cluster} Feature Profile")
                st.pyplot(fig)

            else:
                num_vars = len(features)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                values = np.concatenate((values, [values[0]]))
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.plot(angles, values, color='teal', linewidth=2)
                ax.fill(angles, values, color='teal', alpha=0.25)
                ax.set_yticklabels([])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(features, fontsize=10)
                plt.title(f"Cluster {cluster} Radar Profile", size=14)
                st.pyplot(fig)

    # Download
    input_df['Predicted_Cluster'] = cluster
    csv = input_df.to_csv(index=False)
    st.download_button(
        label="💾 Download Prediction CSV",
        data=csv,
        file_name='Customer_Segmentation_Prediction.csv',
        mime='text/csv'
    )

else:
    st.warning("👆 Please fill in the customer details and click **Predict Customer Cluster** to continue.")

# -----------------------------
# 📊 Visualization Section
# -----------------------------
st.markdown("---")
st.subheader("📈 Customer Segmentation Insights")

uploaded_file = st.file_uploader("📂 Upload full dataset (CSV) for visualization", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    if 'Cluster' not in df.columns:
        with st.spinner("Assigning clusters using trained KMeans model..."):
            scaled = scaler.transform(df[scaler.feature_names_in_])
            df['Cluster'] = kmeans.predict(scaled)

    # Cluster Distribution
    st.markdown("### 🔹 Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
            autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax1.axis('equal')
    st.pyplot(fig1)

    # PCA 2D Visualization
    st.markdown("### 🔹 Cluster Separation (2D PCA Projection)")
    pca = PCA(n_components=2)
    scaled_data = scaler.transform(df[scaler.feature_names_in_])
    pca_data = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = df['Cluster']

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for cluster_id in sorted(df_pca['Cluster'].unique()):
        subset = df_pca[df_pca['Cluster'] == cluster_id]
        ax2.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cluster_id}', s=50)
    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    ax2.set_title('Customer Clusters in 2D PCA Space')
    ax2.legend()
    st.pyplot(fig2)

    # Radar Chart
    st.markdown("### 🔹 Average Feature Profile (Radar Chart)")
    cluster_means = df.groupby('Cluster').mean().reset_index()
    selected_cluster = st.selectbox("Select a cluster to view its feature profile:", cluster_means['Cluster'].unique())
    selected_data = cluster_means[cluster_means['Cluster'] == selected_cluster].iloc[:, 1:]
    features = selected_data.columns
    values = selected_data.values.flatten()
    values = np.concatenate((values, [values[0]]))
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax3.plot(angles, values, color='teal', linewidth=2)
    ax3.fill(angles, values, color='teal', alpha=0.25)
    ax3.set_yticklabels([])
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(features, fontsize=9)
    plt.title(f"Cluster {selected_cluster} Average Feature Profile", size=13)
    st.pyplot(fig3)

    # Heatmap
    st.markdown("### 🔹 Feature Variance Across Clusters (Importance Heatmap)")
    feature_variance = df.groupby('Cluster').mean().T
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(feature_variance, cmap='coolwarm', annot=False, ax=ax4)
    plt.title('Feature Differences Across Clusters')
    st.pyplot(fig4)
else:
    st.info("📤 Upload your customer dataset to explore segmentation visualizations.")
