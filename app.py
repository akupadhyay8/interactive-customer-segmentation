import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib

# Try to import matplotlib; if unavailable, skip plots
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    MATPLOTLIB_AVAILABLE = False

# -- App configuration --
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🛍️ Customer Segmentation Dashboard")

# -- Sidebar navigation --
tab = st.sidebar.selectbox("Choose Activity", ["Segmentation Explorer", "Cluster Predictor"])

# -- Default file paths --
DEFAULT_DATA_PATH = r"E:\Python\Mall Customer Segmentation\Dataset\Mall_Customers.csv"
DEFAULT_MODEL_PATH = r"E:\Python\Mall Customer Segmentation\kmeans_model.joblib"

# -- Utility functions --
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# -- Segmentation Explorer Tab --
if tab == "Segmentation Explorer":
    st.header("1. Data Upload & K-Means Clustering")
    uploaded_file = st.file_uploader(
        "Upload CSV (must include 'Annual Income (k$)' and 'Spending Score (1-100)')", type="csv"
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info(f"No file uploaded. Loading default data from {DEFAULT_DATA_PATH}.")
        df = pd.read_csv(DEFAULT_DATA_PATH)

    required_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Your dataset must include columns: {required_cols}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Feature extraction
    X = df[["Annual Income (k$)", "Spending Score (1-100)" ]].values

    # Cluster count selection
    k = st.slider("Select number of clusters (k)", 2, 10, 5)

    # Fit KMeans
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    df['Cluster'] = labels

    # Elbow method
    if st.checkbox("Show Elbow Method (WCSS)"):
        wcss = [KMeans(n_clusters=i, random_state=42).fit(X).inertia_ for i in range(1, 11)]
        if MATPLOTLIB_AVAILABLE:
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(range(1, 11), wcss, marker='o')
            ax_elbow.set_title('Elbow Method')
            ax_elbow.set_xlabel('Number of clusters')
            ax_elbow.set_ylabel('WCSS')
            st.pyplot(fig_elbow)
        else:
            st.line_chart(pd.DataFrame({'WCSS': wcss}, index=range(1,11)))

    # Cluster visualization
    st.subheader(f"K-Means Clustering Result (k={k})")
    if MATPLOTLIB_AVAILABLE:
        fig_clusters, ax_clusters = plt.subplots(figsize=(8, 6))
        for cluster_num in range(k):
            ax_clusters.scatter(
                X[labels == cluster_num, 0],
                X[labels == cluster_num, 1],
                s=50,
                label=f"Cluster {cluster_num+1}",
            )
        ax_clusters.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            s=200,
            c='black',
            marker='X',
            label='Centroids',
        )
        ax_clusters.set_xlabel('Annual Income (k$)')
        ax_clusters.set_ylabel('Spending Score (1-100)')
        ax_clusters.legend()
        st.pyplot(fig_clusters)
    else:
        scatter_df = pd.DataFrame(X, columns=['Income','Score'])
        scatter_df['Cluster'] = labels
        for c in range(k):
            st.write(f"**Cluster {c+1}**")
            st.write(scatter_df[scatter_df['Cluster']==c].head())

    # Cluster profiling
    st.subheader("Cluster Profiles")
    profile = df.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        Avg_Income=('Annual Income (k$)', 'mean'),
        Avg_Spend=('Spending Score (1-100)', 'mean')
    ).round(2)
    st.table(profile)

    # Download clustered data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download clustered data as CSV",
        data=csv,
        file_name='clustered_customers.csv',
        mime='text/csv'
    )

# -- Cluster Predictor Tab --
elif tab == "Cluster Predictor":
    st.header("2. Predict New Customer Cluster")
    try:
        kmeans_model = load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        st.error(
            f"Could not load model from {DEFAULT_MODEL_PATH}: {e}\n"
            "Please ensure 'kmeans_model.joblib' exists at that location."
        )
    else:
        st.subheader("Enter new customer details:")
        income = st.number_input("Annual Income (k$)", min_value=0)
        score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)
        if st.button("Predict Cluster"):
            label = kmeans_model.predict([[income, score]])[0] + 1
            st.success(f"This customer belongs to **Cluster {label}**.")
