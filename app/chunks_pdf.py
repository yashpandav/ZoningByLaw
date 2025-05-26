import requests
import json
from qdrant_client import QdrantClient
import os
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def extract_heading_and_level(text):
    """
    Extract heading number and title, and determine the level.
    Returns (heading_number, heading_title, level) or (None, None, None)
    """
    # Match e.g. 150.7.60 Ancillary Building Requirements...
    m = re.match(r'^((?:\\d+\\.)+\\d+)\\s+(.+)$', text.strip())
    if m:
        return m.group(1), m.group(2), m.group(1).count('.') + 1
    # Match e.g. (1) Parts of a Garden Suite...
    m = re.match(r'^\\((\\d+)\\)\\s+(.+)$', text.strip())
    if m:
        return m.group(1), m.group(2), 99  # Use 99 for sub-clauses
    return None, None, None

def load_and_split_pdf(pdf_path, chunk_size=800, chunk_overlap=200):
    """Load and split PDF into chunks"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text = splitter.split_documents(documents=documents)
    return text

def get_jina_embeddings(texts, jina_api_key, model="jina-embeddings-v3", task="text-matching"):
    """Get embeddings from Jina AI API"""
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {jina_api_key}'
    }
    data = {
        "model": model,
        "task": task,
        "input": texts
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    result = response.json()
    return [item['embedding'] for item in result['data']]

def init_qdrant_collection(qdrant_client, collection_name, vector_size, distance_metric=Distance.COSINE):
    """Initialize Qdrant collection if it doesn't exist"""
    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating collection '{collection_name}'.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_metric)
        )

def upsert_embeddings_to_qdrant(qdrant_client, collection_name, embeddings, texts):
    """Upsert embeddings and texts to Qdrant"""
    points = []
    hierarchy_stack = []
    last_main_heading = ""
    last_sub_heading = ""
    current_heading_number = ""
    current_heading_title = ""
    current_hierarchy = ""

    for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
        content = text.page_content if hasattr(text, "page_content") else text
        heading_number, heading_title, level = extract_heading_and_level(content)

        # If this chunk starts with a heading, update the current heading/hierarchy
        if level and level < 99:
            if level == 1:
                hierarchy_stack = [heading_title]
            elif level == 2:
                if len(hierarchy_stack) < 1:
                    hierarchy_stack = ["", heading_title]
                else:
                    hierarchy_stack = [hierarchy_stack[0], heading_title]
            elif level == 3:
                while len(hierarchy_stack) < 2:
                    hierarchy_stack.append("")
                hierarchy_stack = [hierarchy_stack[0], hierarchy_stack[1], heading_title]
            last_main_heading = heading_title
            current_heading_number = heading_number
            current_heading_title = heading_title
            current_hierarchy = " > ".join([h for h in hierarchy_stack if h])
            last_sub_heading = ""
        elif level == 99:
            last_sub_heading = heading_title
            if current_hierarchy:
                current_hierarchy = f"{current_hierarchy} > {last_sub_heading}"
        # If not a heading, just use the last known heading/hierarchy
        metadata_dict = {
            "heading_number": current_heading_number or "",
            "heading": current_heading_title or "",
            "hierarchy": current_hierarchy or "",
        }
        points.append({
            "id": idx,
            "vector": embedding,
            "payload": {"text": content, "metadata": metadata_dict}
        })
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Upserted {len(points)} points to collection '{collection_name}'.")

def search_similar_texts(qdrant_client, collection_name, query, jina_api_key, top_k=3):
    """Search for similar texts in Qdrant"""
    query_embedding = get_jina_embeddings([query], jina_api_key)[0]

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k
    )

    print(f"Result Of {query} : {results}")

    return results

def initialize_database(pdf_path, collection_name="jina_embeddings_collection2"):
    """Initialize the database with PDF content"""
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY environment variable is not set")

    # Initialize Qdrant client
    qdrant = QdrantClient(host="localhost", port=6333)

    # Load and split PDF
    print("Loading and splitting PDF...")
    texts = load_and_split_pdf(pdf_path)

    # Generate embeddings
    print("Generating embeddings...")
    raw_texts = [doc.page_content for doc in texts]
    embeddings = get_jina_embeddings(raw_texts, JINA_API_KEY)
    vector_size = len(embeddings[0])

    # Create collection
    init_qdrant_collection(qdrant, collection_name, vector_size)

    # Upload embeddings
    print("Uploading embeddings to Qdrant...")
    upsert_embeddings_to_qdrant(qdrant, collection_name, embeddings, texts)
    print("Database initialization complete!")

if __name__ == "__main__":
    # Example usage for initializing the database
    PDF_PATH = "../GardenSuits.pdf"
    initialize_database(PDF_PATH)
