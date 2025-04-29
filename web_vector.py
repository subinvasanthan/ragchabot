import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import time
import numpy as np
import faiss
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# ========== CONFIGURATION ==========
load_dotenv()
WEBSITE_URL = "https://subinvasanthan.github.io/"  # Website to scrape
MAX_DEPTH = 3  # How many link levels to follow
MAX_PAGES = 50  # Maximum pages to scrape
CHUNK_SIZE = 500  # Size of text chunks
CHUNK_OVERLAP = 50  # Overlap between chunks
DELAY = 1  # Seconds between requests
VECTOR_DB_DIR = "vector_db"  # Directory for vector storage
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ========== INITIALIZATION ==========
USER_AGENT = "Mozilla/5.0 (compatible; RAGBotScraper/1.0)"
HEADERS = {"User-Agent": USER_AGENT}
BLACKLIST = ['.pdf', '.jpg', '.webp', '.png', '.gif', '.zip', '.exe']
EMBEDDING_MODEL = OpenAIEmbeddings()
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)


# ========== CORE FUNCTIONS ==========
def is_valid_url(url, domain):
    """Check if URL should be crawled"""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    if not parsed.netloc.endswith(domain):
        return False
    if any(ext in url.lower() for ext in BLACKLIST):
        return False
    return True


def clean_text(text):
    """Clean and normalize scraped text"""
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = re.sub(r'\n+', '\n', text)  # Collapse newlines
    return text.strip()


def scrape_page(url):
    """Scrape text content from a single page"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()

        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.body
        if main_content:
            text = main_content.get_text(separator='\n')
            return clean_text(text)
        return None

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None


def chunk_content(content, url):
    """Split content into chunks with metadata"""
    chunks = TEXT_SPLITTER.split_text(content)
    return [{
        "text": chunk,
        "source": url,
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


def create_vector_db(embeddings, chunks):
    """Create and save FAISS vector database"""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # Save index and metadata
    timestamp = int(time.time())
    domain = urlparse(WEBSITE_URL).netloc
    db_file = os.path.join(VECTOR_DB_DIR, f"{domain}_vector_db_{timestamp}")

    # Save FAISS index
    faiss.write_index(index, f"{db_file}.index")

    # Save metadata
    metadata = {
        "website": WEBSITE_URL,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with open(f"{db_file}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved vector database to {db_file}.index")
    print(f"Saved metadata to {db_file}_metadata.json")

    return index


def crawl_process_embed():
    """End-to-end website crawling and vector DB creation"""
    domain = urlparse(WEBSITE_URL).netloc
    visited = set()
    queue = [(WEBSITE_URL, 0)]  # (url, depth)
    all_chunks = []
    page_count = 0

    print(f"\nStarting scrape of {WEBSITE_URL} (max depth: {MAX_DEPTH})")

    # Phase 1: Scrape and chunk website content
    with tqdm(desc="Scraping and chunking pages") as pbar:
        while queue and page_count < MAX_PAGES:
            url, depth = queue.pop(0)

            if url in visited or depth > MAX_DEPTH:
                continue

            visited.add(url)
            print(f"\nProcessing: {url} (depth {depth})")

            content = scrape_page(url)
            if content:
                chunks = chunk_content(content, url)
                all_chunks.extend(chunks)
                page_count += 1
                pbar.update(1)

                # Get links for further crawling
                if depth < MAX_DEPTH:
                    try:
                        response = requests.get(url, headers=HEADERS, timeout=5)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(url, href)
                            if is_valid_url(full_url, domain) and full_url not in visited:
                                queue.append((full_url, depth + 1))
                    except:
                        continue

            time.sleep(DELAY)

    if not all_chunks:
        print("No content scraped. Exiting.")
        return

    # Phase 2: Create embeddings
    print("\nGenerating embeddings...")
    chunks_with_embeddings, embeddings = create_embeddings(all_chunks)

    # Phase 3: Create vector database
    print("Creating vector database...")
    create_vector_db(embeddings, chunks_with_embeddings)

    print("\nProcess completed successfully!")


# ========== EXECUTION ==========
if __name__ == "__main__":
    crawl_process_embed()