# clustering.py
# Name: Nathaniel Monney
# Index Number: 10211100300

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

def clustering_page():
    # Minimal CSS for page-specific elements
    st.markdown("""
    <style>
    /* Ensure compatibility with global CSS */
    .module-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        line-height: 1.2;
    }

    @media (max-width: 768px) {
        .module-header {
            font-size: 2.2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="app-container">
        <div style="display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:1rem">
            <i class="material-icons" style="font-size:2.5rem;">group</i>
            <h1 class="module-header">Clustering Module</h1>
        </div>
        <p class="header-subtitle">
            Upload a dataset to perform K-Means clustering and visualize the results interactively.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File upload
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3><i class="material-icons">upload_file</i> Upload Dataset</h3>', unsafe_allow_html=True)
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV file for clustering", type=["csv"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        # Read and preview dataset
        df = pd.read_csv(uploaded_file)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="material-icons">table_chart</i> Dataset Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df.head())
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature selection
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="material-icons">tune</i> Clustering Parameters</h3>', unsafe_allow_html=True)
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = st.multiselect("Select features for clustering", numeric_columns, default=numeric_columns[:2])
            
            if len(selected_features) >= 2:
                X = df[selected_features]
                n_clusters = st.slider("Select number of clusters", 2, 10, 3, help="Number of clusters for K-Means")
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X)
                
                # Visualize clusters
                st.markdown('<h3><i class="material-icons">scatter_plot</i> Clustering Results</h3>', unsafe_allow_html=True)
                if len(selected_features) == 2:
                    fig = px.scatter(df, x=selected_features[0], y=selected_features[1], color='Cluster',
                                    title="K-Means Clustering Results")
                    centroids = kmeans.cluster_centers_
                    fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                                   marker=dict(size=15, color='#dc2626', symbol='x'), name='Centroids')
                    fig.update_layout(
                        plot_bgcolor='#ffffff',
                        paper_bgcolor='#ffffff',
                        font=dict(color='#1e2a44'),
                        title_font=dict(size=20, color='#1e2a44'),
                        xaxis=dict(gridcolor='#e5e7eb'),
                        yaxis=dict(gridcolor='#e5e7eb')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif len(selected_features) >= 3:
                    fig = px.scatter_3d(df, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                                       color='Cluster', title="K-Means Clustering Results (3D)")
                    centroids = kmeans.cluster_centers_
                    fig.add_scatter3d(x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
                                     mode='markers', marker=dict(size=5, color='#dc2626', symbol='x'), name='Centroids')
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(backgroundcolor='#ffffff', gridcolor='#e5e7eb'),
                            yaxis=dict(backgroundcolor='#ffffff', gridcolor='#e5e7eb'),
                            zaxis=dict(backgroundcolor='#ffffff', gridcolor='#e5e7eb')
                        ),
                        font=dict(color='#1e2a44'),
                        title_font=dict(size=20, color='#1e2a44')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Download clustered dataset
                csv = df.to_csv(index=False)
                st.download_button("Download Clustered Dataset", csv, "clustered_data.csv", "text/csv", use_container_width=True)
            else:
                st.warning("Please select at least two features for clustering.")
            st.markdown('</div>', unsafe_allow_html=True)