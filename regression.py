# regression.py
#Name : Nathaniel Monney
#Index Nunber : 10211100300

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

def regression_page():
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
            <i class="material-icons" style="font-size:2.5rem;">trending_up</i>
            <h1 class="module-header">Regression Module</h1>
        </div>
        <p class="header-subtitle">
            Upload a dataset to perform linear regression and predict continuous variables with interactive visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File upload
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3><i class="material-icons">upload_file</i> Upload Dataset</h3>', unsafe_allow_html=True)
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV file for regression", type=["csv"], label_visibility="collapsed")
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

        # Target and feature selection
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="material-icons">tune</i> Model Parameters</h3>', unsafe_allow_html=True)
            target_column = st.selectbox("Select the target column (continuous variable)", df.columns)
            feature_columns = [col for col in df.columns if col != target_column]
            selected_features = st.multiselect("Select feature columns", feature_columns, default=feature_columns[:2])

            # Data preprocessing options
            st.markdown('<h4>Preprocessing Options</h4>', unsafe_allow_html=True)
            drop_na = st.checkbox("Drop missing values", value=True)
            if drop_na:
                df = df.dropna()

            if selected_features:
                X = df[selected_features]
                y = df[target_column]
                test_size = st.slider("Test split size", 0.1, 0.5, 0.2, 0.05, help="Proportion of data for testing")

                # Train the model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Model performance
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.markdown('<h3><i class="material-icons">assessment</i> Model Performance</h3>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                col2.metric("R² Score", f"{r2:.2f}")

                # Visualization: Scatter plot of predictions vs actual
                st.markdown('<h3><i class="material-icons">scatter_plot</i> Predictions vs Actual</h3>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions vs Actual',
                                        marker=dict(color='#4f46e5')))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                        mode='lines', name='Ideal Fit', line=dict(color='#dc2626', dash='dash')))
                fig.update_layout(
                    title="Predictions vs Actual Values",
                    xaxis_title="Actual Values", yaxis_title="Predicted Values",
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(color='#1e2a44'),
                    title_font=dict(size=20, color='#1e2a44'),
                    xaxis=dict(gridcolor='#e5e7eb'),
                    yaxis=dict(gridcolor='#e5e7eb')
                )
                st.plotly_chart(fig, use_container_width=True)

                # If only one feature, show regression line
                if len(selected_features) == 1:
                    st.markdown('<h3><i class="material-icons">show_chart</i> Regression Line</h3>', unsafe_allow_html=True)
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(x=X_test[selected_features[0]], y=y_test, mode='markers',
                                                name='Actual', marker=dict(color='#4f46e5')))
                    fig_line.add_trace(go.Scatter(x=X_test[selected_features[0]], y=y_pred, mode='lines',
                                                name='Regression Line', line=dict(color='#dc2626')))
                    fig_line.update_layout(
                        title="Regression Line",
                        xaxis_title=selected_features[0], yaxis_title=target_column,
                        plot_bgcolor='#ffffff',
                        paper_bgcolor='#ffffff',
                        font=dict(color='#1e2a44'),
                        title_font=dict(size=20, color='#1e2a44'),
                        xaxis=dict(gridcolor='#e5e7eb'),
                        yaxis=dict(gridcolor='#e5e7eb')
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

                # Custom prediction
                st.markdown('<h3><i class="material-icons">prediction</i> Make a Prediction</h3>', unsafe_allow_html=True)
                custom_inputs = {}
                for feature in selected_features:
                    custom_inputs[feature] = st.number_input(f"Enter value for {feature}", value=float(X[feature].mean()))
                if st.button("Predict", use_container_width=True):
                    custom_data = pd.DataFrame([custom_inputs])
                    prediction = model.predict(custom_data)
                    st.markdown(f'<div class="card"><p>Predicted {target_column}: <b>{prediction[0]:.2f}</b></p></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)