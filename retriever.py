# retriever.py
#Name : Nathaniel Monney
#Index Nunber : 10211100300

import numpy as np

class ElectionDataRetriever:
    def __init__(self, embedder):
        """Initialize with a TextEmbedder instance"""
        self.embedder = embedder
        self.chunks = None
        
    def setup_from_chunks(self, chunks):
        """Set up retriever with text chunks"""
        try:
            self.chunks = chunks
            self.embedder.setup_from_chunks(chunks)
            return self
        except Exception as e:
            raise ValueError(f"Error setting up retriever from chunks: {str(e)}")
    
    def setup_from_saved(self, save_dir):
        """Load vector store and chunks from saved directory"""
        try:
            self.embedder.load_vector_store(save_dir)
            self.chunks = self.embedder.chunks
            if self.chunks is None:
                raise ValueError("No chunks found in saved vector store.")
            return self
        except Exception as e:
            raise ValueError(f"Error setting up retriever from saved store: {str(e)}")
    
    def retrieve(self, query, k=5):
        """Retrieve top k relevant chunks for a query"""
        try:
            if self.embedder.vector_store is None:
                raise ValueError("Vector store not initialized.")
            if self.chunks is None:
                raise ValueError("Chunks not loaded.")
                
            query_embedding = self.embedder.create_embeddings([query])[0]
            distances, indices = self.embedder.vector_store.search(np.array([query_embedding]), k)
            
            retrieved = [
                {
                    "text": self.chunks[i]["text"],
                    "score": 1 / (1 + d),  # Convert distance to similarity score
                    "metadata": self.chunks[i]["metadata"]
                }
                for i, d in zip(indices[0], distances[0])
                if i < len(self.chunks)
            ]
            return retrieved
        except Exception as e:
            raise ValueError(f"Error retrieving chunks: {str(e)}")
    
    def format_for_llm(self, chunks):
        """Format retrieved chunks for LLM input"""
        try:
            return "\n\n".join([chunk["text"] for chunk in chunks])
        except Exception as e:
            raise ValueError(f"Error formatting chunks for LLM: {str(e)}")