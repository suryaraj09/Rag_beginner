from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"

def delete_sample_data():
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Load the existing vector store
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Get all the data to find chunks from sample.txt
    data = vector_store.get(include=['metadatas'])
    ids_to_delete = []
    
    # Target file to remove
    target_source = "data\\sample.txt"
    
    for i, metadata in enumerate(data['metadatas']):
        if metadata.get('source') == target_source:
            ids_to_delete.append(data['ids'][i])
            
    if ids_to_delete:
        print(f"Found {len(ids_to_delete)} chunks from '{target_source}'. Deleting...")
        vector_store.delete(ids=ids_to_delete)
        print("Successfully deleted sample data chunks.")
    else:
        print(f"No chunks found from '{target_source}'.")

if __name__ == "__main__":
    if os.path.exists(CHROMA_PATH):
        delete_sample_data()
    else:
        print(f"Directory '{CHROMA_PATH}' not found.")
