import requests
import json
import os
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from one import generate_query_embedding

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

def extract_heading_and_hierarchy(text):
    """Extracts heading number, title, and level based on zoning format"""
    cleaned_text = text.strip()
    
    patterns = [
        r'^(150\.7(?:\.\d+)*)\s+(.+?)(?:\n|$)',
        r'^(150\.7(?:\.\d+)*)\s+(.+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned_text, re.MULTILINE)
        if match:
            code = match.group(1)
            title = match.group(2).strip()
            title = re.split(r'\n', title)[0].strip()
            level = code.count('.')
            return code, title, level
    
    return None, None, None

def build_hierarchical_structure(texts):
    """Build a hierarchical structure from all texts first"""
    hierarchy_map = {}
    
    for doc in texts:
        content = doc.page_content
        heading_code, heading_title, level = extract_heading_and_hierarchy(content)
        
        if heading_code and heading_title:
            hierarchy_map[heading_code] = {
                'title': heading_title,
                'level': level,
                'content': content
            }
    
    return hierarchy_map

def get_full_hierarchy_path(code, hierarchy_map):
    """Get the full hierarchy path for a given code"""
    if not code:
        return ""
    
    path_parts = []
    current_code = code
    
    while current_code:
        if current_code in hierarchy_map:
            path_parts.append(hierarchy_map[current_code]['title'])
        
        if '.' in current_code:
            current_code = '.'.join(current_code.split('.')[:-1])
        else:
            break
    
    path_parts.reverse()
    return " > ".join(path_parts)

def get_embeddings(texts, api_key):
    """Get embeddings from Google Generative AI"""
    genai.configure(api_key=api_key)
    print("Generating embeddings...")
    embeddings = []
    for text in texts:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            print(f"Error generating embedding for text: {e}")
            embeddings.append([0.0] * 768)
    
    return embeddings

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
    """Upsert embeddings and text with metadata into Qdrant"""
    hierarchy_map = build_hierarchical_structure(texts)
    
    points = []
    current_hierarchy_stack = []
    
    for idx, (embedding, doc) in enumerate(zip(embeddings, texts)):
        content = doc.page_content
        heading_code, heading_title, level = extract_heading_and_hierarchy(content)
        
        full_hierarchy = ""
        if heading_code:
            full_hierarchy = get_full_hierarchy_path(heading_code, hierarchy_map)
        else:
            if current_hierarchy_stack:
                full_hierarchy = " > ".join(current_hierarchy_stack)
        
        if heading_code and heading_title:
            current_hierarchy_stack = current_hierarchy_stack[:level]
            current_hierarchy_stack.append(heading_title)
        
        points.append({
            "id": idx,
            "vector": embedding,
            "payload": {
                "text": content,
                "heading_code": heading_code or "",
                "title": heading_title or "",
                "hierarchy": full_hierarchy,
                "level": level if level is not None else 0,
                "page": getattr(doc, 'metadata', {}).get('page', 0),
                "source": getattr(doc, 'metadata', {}).get('source', ''),
                "chunk_index": idx
            }
        })

    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Upserted {len(points)} points to collection '{collection_name}'.")
    return len(points)

def search_similar_texts(qdrant_client, collection_name, query, api_key, top_k=3):
    """Search for similar texts in Qdrant"""

    print("seach....")
    query_embeddings = generate_query_embedding(query)

    # query_embedding = query_embeddings[0]

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embeddings,
        limit=top_k,
        with_payload=True
    )

    return results

def initialize_database(pdf_path, collection_name="embeddings_collection" , qdrant_client=None, api_key=None):
    """Initialize the database with PDF content"""
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")

    # Initialize Qdrant client
    # qdrant_client = QdrantClient(host="localhost", port=6333)
    print("Loading and splitting PDF...")
    texts = load_and_split_pdf(pdf_path)
    print(f"Loaded {len(texts)} text chunks")

    print("Generating embeddings...")
    raw_texts = [doc.page_content for doc in texts]
    embeddings = get_embeddings(raw_texts, api_key)

    if embeddings:
        vector_size = len(embeddings[0])
        print(f"Generated embeddings with dimension: {vector_size}")
    else:
        raise ValueError("Failed to generate embeddings")

    init_qdrant_collection(qdrant_client, collection_name, vector_size)

    print("Uploading embeddings to Qdrant...")
    upsert_embeddings_to_qdrant(qdrant_client, collection_name, embeddings, texts)

    print("Database initialization complete!")
    return qdrant_client, collection_name