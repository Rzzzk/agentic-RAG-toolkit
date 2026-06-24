import os
import sys
from pypdf import PdfReader
from dotenv import load_dotenv

# Ensure Python can find the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the ComponentFactory and environment variables
from src.utils.factory import ComponentFactory
import yaml

def main():
    load_dotenv()
    
    pdf_path = "sample.pdf" # Make sure this file exists in your project root
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return

    # Load config to dynamically instantiate ONLY the vector DB
    with open("configs/free_pdf_bot.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    print("Loading Vector Database...")
    db = ComponentFactory.create_instance(config["vector_db"])

    print(f"Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append(text)

    print(f"Embedding and storing {len(chunks)} pages. This might take a moment...")
    metadatas = [{"source": pdf_path, "page": i+1} for i in range(len(chunks))]
    db.add_documents(documents=chunks, metadatas=metadatas)
    
    print("Ingestion complete! The database is ready.")

if __name__ == "__main__":
    main()