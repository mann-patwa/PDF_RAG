from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_path):
    """
    Loads a PDF and splits it into manageable chunks.
    
    Args:
        pdf_path (str): The file path to the PDF.
        
    Returns:
        list: A list of 'Document' objects, each representing a chunk.
    """
    print(f"Loading PDF from: {pdf_path}")
    
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    if not pages:
        print("Could not load any pages from the PDF.")
        return []
        
    print(f"Loaded {len(pages)} pages.")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    print("Splitting pages into chunks...")
    chunks = text_splitter.split_documents(pages)
    
    print(f"Split the document into {len(chunks)} chunks.")
    
    if chunks:
        print("\n--- Example Chunk (first 100 chars) ---")
        print(chunks[0].page_content[:100] + "...")
        print(f"Metadata (source): {chunks[0].metadata}")

    return chunks


from llama_index.core.schema import Document as LlamaDocument 
from llama_index.core import Settings, VectorStoreIndex, StorageContext

    
def convert_langchain_docs_to_llama(lc_docs):
    """
    Converts a list of LangChain chunks  to LlamaIndex Document objects.
    
    Args:
        lc_docs (list): List of LangChain Documents.
        
    Returns:
        list: List of LlamaIndex Documents.
    """
    print(f"Converting {len(lc_docs)} LangChain docs to LlamaIndex format...")
    llama_docs = []
    for lc_doc in lc_docs:
        llama_doc = LlamaDocument(
            text=lc_doc.page_content,
            metadata=lc_doc.metadata or {}  
        )
        llama_docs.append(llama_doc)
    print("Conversion complete.")
    return llama_docs
    
import torch 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
def setup_llama_index_settings():
    
    print("Setting up LlamaIndex settings for Hugging Face...")
    
    try:
    
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        Settings.embed_model = embed_model
        print("Hugging Face Embedding model configured (BAAI/bge-small-en-v1.5).")
        
        print("LlamaIndex Settings configured successfully.")
        return True
        
    except Exception as e:
        print(f"Error setting up LlamaIndex settings: {e}")
        return False
    
import chromadb # Import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore # Import the integration
from llama_index.core import VectorStoreIndex, StorageContext
def create_vector_index(documents):
    """
    Creates and persists a LlamaIndex VectorStoreIndex using ChromaDB.
    
    Args:
        documents (list): A list of LlamaIndex Document objects.
        
    Returns:
        VectorStoreIndex: The created index.
    """
    print("Initializing ChromaDB...")
    
    db = chromadb.PersistentClient(path="./chroma_db")
    
    chroma_collection = db.get_or_create_collection("hospital_collection")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Creating vector index and storing embeddings...")
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True 
    )
    
    print("Index created and persisted in './chroma_db'.")
    return index

pdf_file_path = "/Users/mannpatwa/Desktop/PDF_RAG/data/HospitalManual.pdf" 
try:
    if setup_llama_index_settings():
        langchain_chunks = load_and_chunk_pdf(pdf_file_path)
        
        if langchain_chunks:
            llama_documents = convert_langchain_docs_to_llama(langchain_chunks)
            
            vector_index = create_vector_index(llama_documents)
            
            print("\n--- Setup Complete! ---")
            print("Your PDF has been loaded, chunked, embedded, and stored in ChromaDB.")
            
       
            
except FileNotFoundError:
    print(f"Error: The file '{pdf_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")