# embedding.py
# Name: Nathaniel Monney
# Index Number: 10211100300


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the SentenceTransformer model"""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_store = None
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.chunks = None
        
    def create_embeddings(self, texts):
        """Create embeddings for a list of texts using SentenceTransformer"""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32
            )
            return embeddings
        except Exception as e:
            raise ValueError(f"Error creating embeddings: {str(e)}")
    
    def setup_from_chunks(self, chunks):
        """Set up FAISS index from text chunks"""
        try:
            self.chunks = chunks
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.create_embeddings(texts)
            
            self.vector_store = faiss.IndexFlatL2(self.dimension)
            self.vector_store.add(embeddings)
            
            return self
        except Exception as e:
            raise ValueError(f"Error setting up vector store: {str(e)}")
    
    def save_vector_store(self, save_dir):
        """Save the FAISS index, model info, and chunks"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            faiss.write_index(self.vector_store, os.path.join(save_dir, "faiss_index.bin"))
            
            with open(os.path.join(save_dir, "embedder_info.pkl"), "wb") as f:
                pickle.dump({
                    "model_name": self.model_name,
                    "dimension": self.dimension,
                    "chunks": self.chunks
                }, f)
        except Exception as e:
            raise ValueError(f"Error saving vector store: {str(e)}")
    
    def load_vector_store(self, save_dir):
        """Load the FAISS index, model info, and chunks"""
        try:
            self.vector_store = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))
            with open(os.path.join(save_dir, "embedder_info.pkl"), "rb") as f:
                info = pickle.load(f)
                self.model_name = info["model_name"]
                self.dimension = info["dimension"]
                self.chunks = info.get("chunks", None)
            return self
        except Exception as e:
            raise ValueError(f"Error loading vector store: {str(e)}")