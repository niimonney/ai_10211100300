# neural_network.py
#Name : Nathaniel Monney
#Index Nunber : 10211100300

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import plotly.graph_objects as go

def neural_network_page():
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
            <i class="material-icons" style="font-size:2.5rem;">neurology</i>
            <h1 class="module-header">Neural Network Module</h1>
        </div>
        <p class="header-subtitle">
            Upload a dataset to train a Feedforward Neural Network for classification and visualize training progress.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File upload
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3><i class="material-icons">upload_file</i> Upload Dataset</h3>', unsafe_allow_html=True)
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV file for classification", type=["csv"], label_visibility="collapsed")
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
            target_column = st.selectbox("Select the target column (categorical)", df.columns)
            feature_columns = [col for col in df.columns if col != target_column]
            selected_features = st.multiselect("Select feature columns", feature_columns, default=feature_columns)

            if selected_features:
                X = df[selected_features]
                y = df[target_column]

                # Preprocessing
                le = LabelEncoder()
                y = le.fit_transform(y)
                num_classes = len(le.classes_)
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Hyperparameters
                st.markdown('<h4>Hyperparameters</h4>', unsafe_allow_html=True)
                epochs = st.slider("Number of epochs", 10, 100, 50, 10)
                learning_rate = st.number_input("Learning rate", 0.0001, 0.1, 0.001, 0.0001)

                # Build and train the model
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
                history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)

                # Plot training history
                st.markdown('<h3><i class="material-icons">show_chart</i> Training Progress</h3>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['accuracy'],
                                       mode='lines', name='Training Accuracy', line=dict(color='#4f46e5')))
                fig.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['val_accuracy'],
                                       mode='lines', name='Validation Accuracy', line=dict(color='#3b82f6')))
                fig.update_layout(
                    title="Training and Validation Accuracy",
                    xaxis_title="Epoch", yaxis_title="Accuracy",
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(color='#1e2a44'),
                    title_font=dict(size=20, color='#1e2a44'),
                    xaxis=dict(gridcolor='#e5e7eb'),
                    yaxis=dict(gridcolor='#e5e7eb')
                )
                st.plotly_chart(fig, use_container_width=True)

                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['loss'],
                                            mode='lines', name='Training Loss', line=dict(color='#4f46e5')))
                fig_loss.add_trace(go.Scatter(x=list(range(epochs)), y=history.history['val_loss'],
                                            mode='lines', name='Validation Loss', line=dict(color='#3b82f6')))
                fig_loss.update_layout(
                    title="Training and Validation Loss",
                    xaxis_title="Epoch", yaxis_title="Loss",
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(color='#1e2a44'),
                    title_font=dict(size=20, color='#1e2a44'),
                    xaxis=dict(gridcolor='#e5e7eb'),
                    yaxis=dict(gridcolor='#e5e7eb')
                )
                st.plotly_chart(fig_loss, use_container_width=True)

                # Evaluate on test data
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                st.markdown('<h3><i class="material-icons">assessment</i> Model Performance</h3>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Test Loss", f"{test_loss:.4f}")
                col2.metric("Test Accuracy", f"{test_accuracy:.4f}")

                # Make predictions
                st.markdown('<h3><i class="material-icons">prediction</i> Make a Prediction</h3>', unsafe_allow_html=True)
                custom_inputs = {}
                for i, feature in enumerate(selected_features):
                    custom_inputs[feature] = st.number_input(f"Enter value for {feature}", value=float(df[feature].mean()))
                if st.button("Predict", use_container_width=True):
                    custom_data = pd.DataFrame([custom_inputs])
                    custom_data = scaler.transform(custom_data)
                    prediction = model.predict(custom_data, verbose=0)
                    predicted_class = le.inverse_transform([np.argmax(prediction[0])])[0]
                    confidence = prediction[0][np.argmax(prediction[0])]
                    st.markdown(f'<div class="card"><p>Predicted Class: <b>{predicted_class}</b> (Confidence: {confidence:.2%})</p></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)