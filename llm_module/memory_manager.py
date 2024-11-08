import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", index_file="faiss_index", metadata_file="metadata.json"):
        self.model = SentenceTransformer(embedding_model)
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.dimension = 384  # Should match the output dimension of the embedding model
        self.index = None  # Initialize without an index
        self.metadata = []

    def faiss_index_exists(self):
        """Check if the FAISS index file exists."""
        return os.path.exists(self.index_file)

    def create_index(self):
        """Create a new FAISS index and save it for future use."""
        # Initialize a new FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []  # Start with an empty metadata list
        
        # Save the newly created index and metadata
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)
        print("New FAISS index and metadata file created.")

    def load_index(self):
        """Load the FAISS index and associated metadata if they exist."""
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)
        print("FAISS index and metadata loaded successfully.")

    def add_memory(self, text, metadata):
        """Embed and add a new memory with metadata to the vector store."""
        embedding = self.embed_text(text)
        if self.index is None:
            self.create_index()  # Create an index if none exists

        self.index.add(np.array([embedding]))
        self.metadata.append(metadata)
        
        # Update the saved index and metadata
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def embed_text(self, text):
        """Embed a single text and return the embedding."""
        embedding = self.model.encode(text)
        return embedding.astype(np.float32)

    def retrieve_memory(self, query, top_k=5):
        """Retrieve top_k relevant memories based on a query."""
        if self.faiss_index_exists(): self.load_index()
        
        if self.index is None or len(self.metadata) == 0:
            print("Memory is empty or index is not initialized.")
            return []

        query_embedding = self.embed_text(query)
        _, indices = self.index.search(np.array([query_embedding]), top_k)
        
        # Retrieve metadata of the top-k indices
        results = [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
        return results
    
    def clear_memory(self):
        """Clear the FAISS index and metadata, resetting the memory."""
        # self.index = faiss.IndexFlatL2(self.dimension)  # Reset FAISS index
        # self.metadata = []  # Clear metadata list

        # Remove files if they exist
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
            
        self.create_index()  # Create a new index and metadata)

        print("Memory cleared: FAISS index and metadata files removed.")

    def check_memory(self):
        """Print the number of stored embeddings and the metadata for inspection."""
        if self.faiss_index_exists():
            self.load_index()
        else:
            print("No memory stored.")
            return

        if self.index is None or len(self.metadata) == 0:
            print("No memory stored.")
        else:
            print(f"Stored embeddings: {len(self.metadata)}")
            print("Metadata of stored interactions:")
            for i, meta in enumerate(self.metadata):
                print(f"{i+1}: User: {meta['user']} - Response: {meta['response']}")
