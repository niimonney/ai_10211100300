# llm_interface.py
#Name : Nathaniel Monney
#Index Nunber : 10211100300

import pandas as pd
import os
import streamlit as st
from data_processor import GhanaElectionDataProcessor
from embedding import TextEmbedder
from retriever import ElectionDataRetriever
from generator import GeminiGenerator
from evaluation import RagEvaluator
from visualization import MultimodalVisualizer
import plotly.express as px
import plotly.graph_objects as go

def rag_system(data_path=None, vector_store_dir="./vector_store", 
              model_id="gemini-1.5-pro", use_saved_vectors=False, api_key=None):
    try:
        if not api_key:
            raise ValueError("Gemini API key is required to initialize the RAG system.")
        
        processor = GhanaElectionDataProcessor()
        embedder = TextEmbedder()
        retriever = ElectionDataRetriever(embedder)
        generator = GeminiGenerator(model_name=model_id)
        evaluator = RagEvaluator()
        
        if data_path:
            raw_data = processor.load_data(data_path)
            processed_data = processor.preprocess_data()
            chunks = processor.create_text_chunks()
            visualizer = MultimodalVisualizer(processed_data)
            
            if use_saved_vectors and os.path.exists(vector_store_dir):
                retriever.setup_from_saved(vector_store_dir)
            else:
                retriever.setup_from_chunks(chunks)
                embedder.save_vector_store(vector_store_dir)
        else:
            if use_saved_vectors and os.path.exists(vector_store_dir):
                retriever.setup_from_saved(vector_store_dir)
                visualizer = MultimodalVisualizer()
            else:
                raise ValueError("Either data_path or use_saved_vectors with existing vectors must be provided")
        
        generator.load_model(api_key)
        generator.setup_pipeline()
        chain = generator.create_rag_chain()
        
        return {
            "processor": processor,
            "retriever": retriever,
            "generator": generator,
            "evaluator": evaluator,
            "visualizer": visualizer,
            "chain": chain
        }
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def ask_question(rag_components, question, k=5):
    try:
        retriever = rag_components["retriever"]
        generator = rag_components["generator"]
        chain = rag_components["chain"]
        evaluator = rag_components["evaluator"]
        
        retrieved_chunks = retriever.retrieve(question, k=k)
        context = retriever.format_for_llm(retrieved_chunks)
        answer = generator.generate_answer(chain, context, question)
        evaluation = evaluator.evaluate_response(question, context, answer)
        
        return answer, retrieved_chunks, evaluation
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return None, None, None

def ghana_election_multimodal_app():
    # Minimal CSS for page-specific elements
    st.markdown("""
    <style>
    /* Ensure compatibility with global CSS */
    .rag-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    @media (max-width: 768px) {
        .rag-header {
            font-size: 2.2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with Icon and Subtitle
    st.markdown("""
    <div class="app-container">
        <div style="display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:1rem">
            <i class="material-icons" style="font-size:2.5rem;">bolt</i>
            <h1 class="rag-header">Ghana Election RAG System</h1>
        </div>
        <p class="header-subtitle">
            Dive into Ghana election data with AI-powered insights and interactive visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration Section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3><i class="material-icons">settings</i> Configuration</h3>', unsafe_allow_html=True)
        
        if 'rag_components' not in st.session_state:
            st.session_state.rag_components = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'question_history' not in st.session_state:
            st.session_state.question_history = []
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
        
        api_key_input = st.text_input("Enter your Gemini API Key:", 
                                    type="password", 
                                    value=st.session_state.api_key,
                                    placeholder="Your API key is secure")
        
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("API Key saved successfully!")
        
        st.markdown("<h4>Data Source</h4>", unsafe_allow_html=True)
        data_option = st.radio("Choose a data source:", ["Upload CSV", "Use Saved Vectors"], 
                              horizontal=True, 
                              label_visibility="collapsed")
        
        if data_option == "Upload CSV" and not st.session_state.data_loaded:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Ghana Election Data CSV", 
                                           type=['csv'], 
                                           key="csv_uploader",
                                           label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            if uploaded_file and st.session_state.api_key:
                with st.spinner("Processing data..."):
                    try:
                        os.makedirs("data", exist_ok=True)
                        data_path = "data/temp_data.csv"
                        with open(data_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.rag_components = rag_system(
                            data_path=data_path,
                            use_saved_vectors=False,
                            api_key=st.session_state.api_key
                        )
                        if st.session_state.rag_components:
                            st.session_state.data_loaded = True
                            st.success("Data processed and RAG system initialized!")
                        else:
                            st.error("Failed to initialize RAG system.")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
            elif not st.session_state.api_key:
                st.warning("Please enter your Gemini API Key.")
        
        elif data_option == "Use Saved Vectors" and not st.session_state.data_loaded:
            vector_dir = st.text_input("Vector Store Directory", 
                                      value="./vector_store",
                                      placeholder="e.g., ./vector_store")
            if st.button("Load Vectors", use_container_width=True):
                if os.path.exists(vector_dir) and st.session_state.api_key:
                    with st.spinner("Loading saved vector store..."):
                        try:
                            st.session_state.rag_components = rag_system(
                                data_path=None,
                                vector_store_dir=vector_dir,
                                use_saved_vectors=True,
                                api_key=st.session_state.api_key
                            )
                            if st.session_state.rag_components:
                                st.session_state.data_loaded = True
                                st.success("Vector store loaded and RAG system initialized!")
                            else:
                                st.error("Failed to initialize RAG system.")
                        except Exception as e:
                            st.error(f"Error loading vectors: {str(e)}")
                elif not st.session_state.api_key:
                    st.warning("Please enter your Gemini API Key.")
                else:
                    st.error(f"Vector store directory {vector_dir} not found!")
        
        if st.session_state.data_loaded:
            if st.button("Reset System", use_container_width=True, type="secondary"):
                st.session_state.rag_components = None
                st.session_state.data_loaded = False
                st.session_state.question_history = []
                st.session_state.api_key = ""
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Content
    if st.session_state.data_loaded and st.session_state.rag_components:
        with st.container():
            tab1, tab2, tab3 = st.tabs(["⚡️ Ask Questions", "📊 Visualizations", "📈 Evaluation"])
            
            with tab1:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown('<h3><i class="material-icons">question_answer</i> Ask About Ghana Elections</h3>', 
                          unsafe_allow_html=True)
                
                question = st.text_input("Your question:", 
                                       placeholder="E.g., Which party won the most votes in 2020?", 
                                       label_visibility="collapsed")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    k_value = st.slider("Number of context chunks", 1, 10, 5, 
                                      help="How many document chunks to retrieve for context",
                                      label_visibility="visible")
                with col2:
                    submit_button = st.button("Ask Question", use_container_width=True, type="primary")
                
                if submit_button and question:
                    with st.spinner("Searching election data and generating answer..."):
                        answer, chunks, eval_result = ask_question(
                            st.session_state.rag_components, 
                            question,
                            k=k_value
                        )
                    
                    if answer:
                        st.session_state.question_history.append({
                            "question": question,
                            "answer": answer,
                            "chunks": chunks,
                            "evaluation": eval_result
                        })
                        
                        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                        st.markdown("<h4>Answer:</h4>", unsafe_allow_html=True)
                        st.write(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        with st.expander("View Retrieved Context"):
                            for i, chunk in enumerate(chunks):
                                st.markdown(f'<div class="context-chunk">', unsafe_allow_html=True)
                                st.markdown(f"**Context {i+1}** (Relevance: {chunk['score']:.4f})")
                                st.write(chunk['text'])
                                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.session_state.question_history:
                    st.markdown('<h4><i class="material-icons">history</i> Recent Questions</h4>', 
                              unsafe_allow_html=True)
                    for i, item in enumerate(reversed(st.session_state.question_history[-5:])):
                        with st.expander(f"Q: {item['question']}"):
                            st.markdown("**Answer:**")
                            st.write(item['answer'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown('<h3><i class="material-icons">bar_chart</i> Election Data Visualizations</h3>', 
                          unsafe_allow_html=True)
                
                if hasattr(st.session_state.rag_components["visualizer"], "data") and \
                   st.session_state.rag_components["visualizer"].data is not None:
                    
                    visualizer = st.session_state.rag_components["visualizer"]
                    
                    st.markdown("<h4>Top Parties by Votes</h4>", unsafe_allow_html=True)
                    party_fig = visualizer.plot_party_votes(top_n=5)
                    if party_fig:
                        party_fig.update_layout(
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff',
                            font=dict(color='#1e2a44'),
                            title_font=dict(size=20, color='#1e2a44'),
                            xaxis=dict(gridcolor='#e5e7eb'),
                            yaxis=dict(gridcolor='#e5e7eb')
                        )
                        st.plotly_chart(party_fig, use_container_width=True)
                    
                    st.markdown("<h4>Vote Distribution by Region</h4>", unsafe_allow_html=True)
                    region_fig = visualizer.plot_regional_distribution()
                    if region_fig:
                        region_fig.update_layout(
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff',
                            font=dict(color='#1e2a44'),
                            title_font=dict(size=20, color='#1e2a44')
                        )
                        st.plotly_chart(region_fig, use_container_width=True)
                    
                    st.markdown("<h4>Party Comparison by Region</h4>", unsafe_allow_html=True)
                    party_region_fig = visualizer.plot_party_comparison_by_region()
                    if party_region_fig:
                        party_region_fig.update_layout(
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff',
                            font=dict(color='#1e2a44'),
                            title_font=dict(size=20, color='#1e2a44'),
                            xaxis=dict(gridcolor='#e5e7eb'),
                            yaxis=dict(gridcolor='#e5e7eb')
                        )
                        st.plotly_chart(party_region_fig, use_container_width=True)
                    
                    st.markdown("<h4>Voter Turnout by Region</h4>", unsafe_allow_html=True)
                    turnout_fig = visualizer.plot_voter_turnout()
                    if turnout_fig:
                        turnout_fig.update_layout(
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff',
                            font=dict(color='#1e2a44'),
                            title_font=dict(size=20, color='#1e2a44'),
                            xaxis=dict(gridcolor='#e5e7eb'),
                            yaxis=dict(gridcolor='#e5e7eb')
                        )
                        st.plotly_chart(turnout_fig, use_container_width=True)
                else:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="text-align:center;padding:2rem">
                        <i class="material-icons" style="font-size:3rem;">info</i>
                        <h4 style="margin-top:1rem">No Visualization Data</h4>
                        <p style="color:#4b5563">Please upload a CSV file with election data to enable visualizations</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown('<h3><i class="material-icons">analytics</i> System Evaluation Metrics</h3>', 
                          unsafe_allow_html=True)
                
                if st.session_state.question_history:
                    evaluator = st.session_state.rag_components["evaluator"]
                    summary = evaluator.generate_summary()
                    
                    st.markdown("<h4>Average Performance</h4>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{summary['average_metrics'].get('context_relevance', 0):.2f}</div>
                            <div class="metric-label">Context Relevance</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{summary['average_metrics'].get('response_completeness', 0):.2f}</div>
                            <div class="metric-label">Response Completeness</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{summary['average_metrics'].get('response_conciseness', 0):.2f}</div>
                            <div class="metric-label">Response Conciseness</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<h4>Recent Evaluations</h4>", unsafe_allow_html=True)
                    for i, item in enumerate(reversed(st.session_state.question_history[-5:])):
                        with st.expander(f"Question: {item['question']}"):
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("Context Relevance", 
                                         f"{item['evaluation']['metrics'].get('context_relevance', 0):.2f}")
                            with cols[1]:
                                st.metric("Response Completeness", 
                                         f"{item['evaluation']['metrics'].get('response_completeness', 0):.2f}")
                            with cols[2]:
                                st.metric("Response Conciseness", 
                                         f"{item['evaluation']['metrics'].get('response_conciseness', 0):.2f}")
                else:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="text-align:center;padding:2rem">
                        <i class="material-icons" style="font-size:3rem;">question_answer</i>
                        <h4 style="margin-top:1rem">No Evaluations Yet</h4>
                        <p style="color:#4b5563">Ask some questions to see evaluation metrics</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    ghana_election_multimodal_app()