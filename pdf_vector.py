import os
import re
import json
import time
import numpy as np
import faiss
from tqdm import tqdm
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# ========== CONFIGURATION ==========
load_dotenv()
PDF_FOLDER = "pdfs"  # Folder containing PDFs
CHUNK_SIZE = 500  # Size of text chunks
CHUNK_OVERLAP = 50  # Overlap between chunks
VECTOR_DB_DIR = "vector_db"  # Directory for vector storage
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ========== INITIALIZATION ==========
EMBEDDING_MODEL = OpenAIEmbeddings()
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)


# ========== CORE FUNCTIONS ==========
def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None


def clean_text(text):
    """Clean and normalize extracted text"""
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = re.sub(r'\n+', '\n', text)  # Collapse newlines
    return text.strip()


def chunk_content(content, source):
    """Split content into chunks with metadata"""
    chunks = TEXT_SPLITTER.split_text(content)
    return [{
        "text": chunk,
        "source": source,
        "chunk_num": i,
        "total_chunks": len(chunks)
    } for i, chunk in enumerate(chunks)]


def create_embeddings(chunks):
    """Generate embeddings for all chunks"""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = EMBEDDING_MODEL.embed_documents(texts)

    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks, embeddings


def create_vector_db(embeddings, chunks, db_name):
    """Create and save FAISS vector database"""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # Save index and metadata
    timestamp = int(time.time())
    db_file = os.path.join(VECTOR_DB_DIR, f"{db_name}_vector_db_{timestamp}")

    # Save FAISS index
    faiss.write_index(index, f"{db_file}.index")

    # Save metadata
    metadata = {
        "source": "pdf_folder",
        "pdf_folder": PDF_FOLDER,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with open(f"{db_file}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved vector database to {db_file}.index")
    print(f"Saved metadata to {db_file}_metadata.json")

    return index


def process_pdfs_to_vector_db():
    """End-to-end PDF processing and vector DB creation"""
    if not os.path.exists(PDF_FOLDER):
        print(f"PDF folder '{PDF_FOLDER}' not found")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{PDF_FOLDER}'")
        return

    all_chunks = []

    # Phase 1: Extract and chunk PDF content
    print(f"\nProcessing {len(pdf_files)} PDFs from '{PDF_FOLDER}'")
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        if text:
            cleaned_text = clean_text(text)
            chunks = chunk_content(cleaned_text, pdf_file)
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No content extracted from PDFs. Exiting.")
        return

    # Phase 2: Create embeddings
    print("\nGenerating embeddings...")
    chunks_with_embeddings, embeddings = create_embeddings(all_chunks)

    # Phase 3: Create vector database
    print("Creating vector database...")
    db_name = os.path.basename(os.path.normpath(PDF_FOLDER))
    create_vector_db(embeddings, chunks_with_embeddings, db_name)

    print("\nPDF processing completed successfully!")


# ========== EXECUTION ==========
if __name__ == "__main__":
    process_pdfs_to_vector_db()