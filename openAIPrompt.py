
import os
import torch
from llama_index.core import Settings, VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq  # REAL Groq integration
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message="A new version of the model card has been uploaded")


def setup_llama_index_settings():
    """Configure LlamaIndex to use HuggingFace embeddings + Groq LLM."""
    print("Setting up LlamaIndex settings for Groq...")

    try:
        # --- Embeddings ---
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        Settings.embed_model = embed_model
        print("HuggingFace Embedding model configured.")

        # --- Groq LLM ---
        groq_api_key = ""

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set!")

        Settings.llm = Groq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            max_tokens=256,
            temperature=0.1,
            system_prompt=(
                "You are a precise and concise Q&A assistant. "
                "Use the provided context to answer. "
                "If the context does not help, say you don't know."
            ),
        )

        print("Groq LLM configured successfully.")
        return True

    except Exception as e:
        print(f"Error setting up LlamaIndex settings: {e}")
        return False


def load_vector_index():
    """Loads an existing vector index from ChromaDB."""
    print("Loading existing index from ChromaDB...")

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("hospital_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    print("Index loaded successfully.")
    return index


if __name__ == "__main__":
    if setup_llama_index_settings():
        try:
            index = load_vector_index()
            query_engine = index.as_query_engine()

            print("\n--- Ready to Query (Groq-powered RAG)! ---")
            print("Type your question and press Enter. Type 'quit' to exit.")

            while True:
                user_query = input("\nQuery: ")
                if user_query.lower() == 'quit':
                    break

                print("Processing query...")

                response = query_engine.query(user_query)

                print("\nResponse:")
                print(str(response))

        except Exception as e:
            print(f"An error occurred during query: {e}")
       

        