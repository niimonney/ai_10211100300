# main.py
#Name : Nathaniel Monney
#Index Nunber : 10211100300

import streamlit as st
from regression import regression_page
from clustering import clustering_page
from neural_network import neural_network_page
from llm_interface import ghana_election_multimodal_app

# Set page config for wide layout and modern theme
st.set_page_config(
    page_title="AI Model Playground",
    page_icon="⚡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS with comprehensive styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

* {
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
}

body {
    background: linear-gradient(135deg, #e6e9f0 0%, #b3c7e6 100%) !important;
    color: #1e2a44;
    margin: 0;
    padding: 0;
}

.app-container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 2.5rem;
}

.stSidebar {
    background: #ffffff;
    border-right: 1px solid #d1d5db;
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.06);
    padding-top: 1.5rem;
}

.stSidebar [data-testid="stSidebarNav"] {
    padding: 1rem;
}

.stSidebar [data-testid="stSidebarNav"] > div {
    border-radius: 8px;
    margin: 0.3rem 0;
    transition: all 0.2s ease;
}

.stSidebar [data-testid="stSidebarNav"] label {
    color: #374151;
    font-weight: 500;
    font-size: 1rem;
    padding: 0.85rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    border-radius: 8px;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.stSidebar [data-testid="stSidebarNav"] label:hover {
    background: #f1f5f9;
    border-color: #4f46e5;
    transform: translateX(4px);
}

.stSidebar [data-testid="stSidebarNav"] input:checked + label {
    background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
    color: #ffffff;
    border: none;
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.3);
}

.material-icons {
    font-size: 1.3rem;
    vertical-align: middle;
    color: #4f46e5;
}

h1, .module-header, .rag-header {
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    line-height: 1.2;
}

h2 {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1e2a44;
    text-align: center;
    margin-bottom: 1rem;
}

h3 {
    color: #1e2a44;
    font-weight: 600;
    font-size: 1.6rem;
}

h4 {
    color: #4b5563;
    font-weight: 500;
    font-size: 1.3rem;
}

.header-subtitle {
    font-size: 1.1rem;
    color: #4b5563;
    text-align: center;
    margin-bottom: 2rem;
    line-height: 1.5;
}

.card, div.home-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
    color: #374151;
    line-height: 1.6;
}

.card:hover, div.home-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.upload-section {
    background: #f9fafb;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 2px dashed #d1d5db;
    transition: border-color 0.3s ease;
}

.upload-section:hover {
    border-color: #4f46e5;
}

.stTextInput > div > div > input, .stSelectbox > div > div > select, .stMultiselect > div > div > div, .stNumberInput > div > div > input, .stSlider > div {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    background: #ffffff;
    color: #1e2a44;
    padding: 0.85rem;
    transition: all 0.2s ease;
}

.stTextInput > div > div > input:focus, .stSelectbox > div > div > select:focus, .stMultiselect > div > div > div:focus, .stNumberInput > div > div > input:focus {
    border-color: #4f46e5;
    box-shadow: 0 0 8px rgba(79, 70, 229, 0.2);
    outline: none;
}

.stButton > button {
    background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.85rem 1.5rem;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
}

.stTabs [data-baseweb="tab-list"] {
    background: #f9fafb;
    border-radius: 12px;
    padding: 0.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.85rem 1.5rem;
    font-weight: 500;
    color: #4b5563;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.3);
}

.stTabs [data-baseweb="tab"]:hover {
    background: #e5e7eb;
    color: #1e2a44;
}

.tab-content {
    background: #ffffff;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    border: 1px solid #e5e7eb;
}

.answer-container {
    background: #f9fafb;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid #4f46e5;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.context-chunk {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #a5b4fc;
}

.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 600;
    color: #4f46e5;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.95rem;
    color: #4b5563;
    font-weight: 500;
}

.stExpander {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    margin-bottom: 1rem;
    background: #ffffff;
}

.stExpander > div > div {
    background: #ffffff;
    border-radius: 8px;
    padding: 1rem;
    color: #1e2a44;
}

.stPlotlyChart {
    background: #ffffff;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}

.stDataFrame {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #ffffff;
}

.stMetric {
    background: #f9fafb;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .app-container {
        padding: 1.5rem;
    }

    h1, .module-header, .rag-header {
        font-size: 2.2rem;
    }

    .card, .tab-content {
        padding: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 1.5rem;'>
    <h2 style='color: #1e2a44; font-weight: 600;'>AI Model Playground</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Home",
        "📈 Regression",
        "🧩 Clustering",
        "🧠 Neural Network",
        "💬 LLM"
    ],
    label_visibility="collapsed"
)

# Page routing
if page == "🏠 Home":
    st.markdown('<div class="app-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="module-header">Welcome to the AI Model Playground</h1>', unsafe_allow_html=True)
    st.markdown('<div class="home-card">', unsafe_allow_html=True)
    st.write("Explore cutting-edge AI and machine learning models in an interactive, user-friendly environment.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
elif page == "📈 Regression":
    regression_page()
elif page == "🧩 Clustering":
    clustering_page()
elif page == "🧠 Neural Network":
    neural_network_page()
elif page == "💬 LLM":
    ghana_election_multimodal_app()