import warnings
import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq

warnings.filterwarnings("ignore", category=UserWarning,
                        message="A new version of the model card has been uploaded")

# Load once at server start
is_initialized = False
query_engine = None


def initialize_rag():
    """Loads LlamaIndex + Groq + Chroma only once."""
    global is_initialized, query_engine

    if is_initialized:
        return query_engine

    print("Initializing Groq-powered RAG pipeline...")

    # Embeddings
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    # LLM
    Settings.llm = Groq(
        api_key="API_KEY",
        model="llama-3.3-70b-versatile",
        max_tokens=256,
        temperature=0.1,
        system_prompt=(
            "You are a precise and concise Q&A assistant. "
            "Use the provided context to answer. If the context does not help, say you don't know."
        ),
    )

    # Vector store load
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("hospital_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    query_engine = index.as_query_engine()

    is_initialized = True
    print("RAG initialized successfully.")

    return query_engine


def ask_llm(query: str) -> str:
    """Ask Groq + LlamaIndex and return answer."""
    engine = initialize_rag()
    response = engine.query(query)
    return str(response)
