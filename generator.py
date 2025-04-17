# generator.py
#Name : Nathaniel Monney
#Index Nunber : 10211100300

import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai

class GeminiGenerator:
    def __init__(self, model_name="gemini-pro"):
        """Initialize with a Gemini model"""
        self.model_name = model_name
        self.api_key = None
        self.gemini_llm = None
        
    def load_model(self, api_key=None):
        """Set up the Gemini API with your API key"""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please provide an API key or set the GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        self.gemini_llm = GoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=512
        )
        
        return self
    
    def setup_pipeline(self):
        """Setup is simplified for Gemini as LangChain handles it directly"""
        if not self.gemini_llm:
            raise ValueError("Model not loaded. Call load_model first.")
        
        return self
    
    def create_rag_chain(self):
        """Create a RAG chain with appropriate prompt engineering for Gemini"""
        if not self.gemini_llm:
            raise ValueError("Model not set up. Call load_model first.")
        
        template = """You are an expert on Ghana election data analysis. Use only the provided context to answer the question accurately. If you don't have enough information based on the context, say that you don't have enough information.

Context:
{context}

Question: {question}
"""
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        chain = LLMChain(
            llm=self.gemini_llm,
            prompt=prompt,
            verbose=True
        )
        
        return chain
    
    def generate_answer(self, chain, context, question):
        """Generate an answer using the RAG chain"""
        result = chain.run(context=context, question=question)
        return result.strip()
    
    def cleanup(self):
        """Cleanup is minimal for API-based models"""
        self.gemini_llm = None