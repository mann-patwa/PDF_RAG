import os
import torch
from llama_index.core import Settings, VectorStoreIndex, StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import warnings

# Suppress a specific UserWarning from transformers
warnings.filterwarnings("ignore", category=UserWarning, 
                        message="A new version of the model card has been uploaded")


def setup_llama_index_settings():
    """Configures the global LlamaIndex settings to use free Hugging Face models."""
    
    print("Setting up LlamaIndex settings for Hugging Face...")
    
    try:
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        Settings.embed_model = embed_model
        print("Hugging Face Embedding model configured.")
        
        system_prompt = (
            "You are a Q&A assistant. Your goal is to answer questions as "
            "accurately as possible based on the context provided. "
            "If the context does not contain the answer, state that."
        )
        query_wrapper_prompt = "<|user|>\n{query_str}\n</s>\n<|assistant|axn"

        llm = HuggingFaceLLM(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            context_window=2048,
            max_new_tokens=256,
            device_map="auto",
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
        )
        Settings.llm = llm
        print("Hugging Face LLM configured.")
        
        print("LlamaIndex Settings configured successfully.")
        return True
        
    except Exception as e:
        print(f"Error setting up LlamaIndex settings: {e}")
        return False
def load_vector_index():
    """
    Loads an existing vector index from ChromaDB.
    
    Returns:
        VectorStoreIndex: The loaded index.
    """
    print("Loading existing index from ChromaDB...")
    
    db = chromadb.PersistentClient(path="./chroma_db")
    
    chroma_collection = db.get_collection("pdf_rag_collection")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
  
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    
    print("Index loaded successfully.")
    return index


if __name__ == "__main__":
    if setup_llama_index_settings():
        try:
            index = load_vector_index()
            
  
            query_engine = index.as_query_engine()
            
            print("\n--- Ready to Query! ---")
            print("Type your question and press Enter. Type 'quit' to exit.")
            
            while True:
                user_query = input("\nQuery: ")
                if user_query.lower() == 'quit':
                    break
                
                print("Processing query... (This may take a moment)")
                
                response = query_engine.query(user_query)
                
                print("\nResponse:")
                print(str(response))
                
        except Exception as e:
            print(f"An error occurred during query: {e}")