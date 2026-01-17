from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"

def peek_vectors():
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Load the existing vector store
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Get all the data from the vector store
    # Note: .get() returns a dictionary with ids, metadatas, documents, and optionally embeddings
    data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
    
    num_chunks = len(data['ids'])
    print(f"Total chunks in DB: {num_chunks}")
    
    if num_chunks > 0:
        # Let's peek at the first chunk
        print("\n--- SAMPLE CHUNK (Chunk 0) ---")
        print(f"ID: {data['ids'][0]}")
        print(f"Metadata: {data['metadatas'][0]}")
        print(f"Document Snippet: {data['documents'][0][:100]}...")
        
        # Show a portion of the actual vector (the embeddings)
        vector = data['embeddings'][0]
        print(f"Vector Length: {len(vector)}")
        print(f"Vector (first 10 numbers): {vector[:10]}...")
        print("------------------------------")

if __name__ == "__main__":
    if os.path.exists(CHROMA_PATH):
        peek_vectors()
    else:
        print(f"Directory '{CHROMA_PATH}' not found. Run ingestion first!")
