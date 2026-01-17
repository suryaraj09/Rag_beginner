import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

# Configuration
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def load_documents():
    """Load documents from the data directory."""
    print("Loading documents...")
    
    docs = []
    # Load files directly to be sure
    for s_file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, s_file)
        if s_file.endswith(".txt"):
            print(f"Loading TXT: {s_file}")
            loader = TextLoader(file_path, autodetect_encoding=True)
            docs.extend(loader.load())
        elif s_file.endswith(".pdf"):
            print(f"Loading PDF: {s_file}")
            loader = PyMuPDFLoader(file_path)
            docs.extend(loader.load())
            
    print(f"Loaded {len(docs)} documents.")
    return docs

def split_documents(documents):
    """Split documents into chunks."""
    print("Splitting documents into chunks...")
    
    # Debug: Check all documents
    for i, doc in enumerate(documents):
        print(f"Debug: Doc {i} content length: {len(doc.page_content)}")
        if len(doc.page_content) > 0:
            print(f"Debug: Doc {i} snippet: {doc.page_content[:50]}...")
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    """Embed chunks and save to Chroma DB."""
    if not chunks:
        print("Error: No chunks to save to Chroma DB.")
        return

    print("Embedding chunks and saving to Chroma DB (this may take a moment)...")
    
    # Initialize Google Gemini Embeddings
    # model="models/text-embedding-004" is the latest embedding model from Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Create the vector store and persist it to the local directory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Successfully saved {len(chunks)} chunks to '{CHROMA_PATH}'.")

def run_ingestion():
    # Ensure data directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created '{DATA_PATH}/' directory. Please place your .txt or .pdf files there.")
        return

    docs = load_documents()
    if not docs:
        print(f"No documents found in '{DATA_PATH}/'. Please add some files and run again.")
        return

    chunks = split_documents(docs)
    save_to_chroma(chunks)

if __name__ == "__main__":
    run_ingestion()