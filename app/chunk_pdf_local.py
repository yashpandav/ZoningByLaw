import requests
import json
import os
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from qdrant_client.http.models import PointStruct
import time

def setup_local_qdrant_storage(storage_path="./qdrant_storage"):
    """
    Setup local Qdrant with disk storage
    
    Args:
        storage_path: Path where Qdrant data will be stored on disk
    
    Returns:
        QdrantClient instance configured for local disk storage
    """
    # Create storage directory if it doesn't exist
    os.makedirs(storage_path, exist_ok=True)
    
    try:
        # Initialize Qdrant client with local disk storage
        qdrant_client = QdrantClient(path=storage_path)
        print(f"✅ Qdrant initialized with disk storage at: {os.path.abspath(storage_path)}")
        
        # List existing collections
        collections = qdrant_client.get_collections()
        if collections.collections:
            print(f"Existing collections: {[c.name for c in collections.collections]}")
        else:
            print("No existing collections found")
            
        return qdrant_client
        
    except Exception as e:
        print(f"❌ Failed to initialize Qdrant with disk storage: {e}")
        return None

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
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
    
    for i, text in enumerate(texts):
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

def generate_query_embedding(query, api_key=None):
    """Generate embedding for a single query"""
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    genai.configure(api_key=api_key)
    
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

def check_collection_exists_and_has_data(qdrant_client, collection_name):
    """Check if collection exists and has data"""
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        points_count = collection_info.points_count
        
        print(f"✅ Collection '{collection_name}' already exists.")
        print(f"   - Points count: {points_count}")
        print(f"   - Vector size: {collection_info.config.params.vectors.size}")
        
        if points_count > 0:
            print(f"🔄 Collection '{collection_name}' already has {points_count} points. Skipping data processing.")
            return True, collection_info
        else:
            print(f"📝 Collection '{collection_name}' exists but is empty. Will populate with data.")
            return False, collection_info
            
    except Exception:
        print(f"📝 Collection '{collection_name}' does not exist. Will create and populate.")
        return False, None

def init_qdrant_collection(qdrant_client, collection_name, vector_size, distance_metric=Distance.COSINE):
    """Initialize Qdrant collection if it doesn't exist"""
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' already exists.")
        return collection_info
    except Exception:
        print(f"📝 Creating collection '{collection_name}' with vector size {vector_size}...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_metric)
        )
        print(f"✅ Collection '{collection_name}' created successfully!")
        return None

def upsert_embeddings_to_qdrant(qdrant_client, collection_name, embeddings, texts):
    """Upsert embeddings and text with metadata into Qdrant"""
    hierarchy_map = build_hierarchical_structure(texts)
    
    points = []
    current_hierarchy_stack = []
    
    print("📝 Preparing data points for upsert...")
    
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
        
        # Create PointStruct object instead of dictionary
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "text": content,
                "heading_code": heading_code or "",
                "title": heading_title or "",
                "hierarchy": full_hierarchy,
                "level": level if level is not None else 0,
                "page": getattr(doc, 'metadata', {}).get('page', 0),
                "source": getattr(doc, 'metadata', {}).get('source', ''),
                "chunk_index": idx
            }
        )
        points.append(point)
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"Prepared {idx + 1}/{len(texts)} points")

    print(f"📤 Upserting {len(points)} points to collection '{collection_name}'...")
    
    # Batch upsert for better performance
    batch_size = 100
    total_batches = (len(points) + batch_size - 1) // batch_size
    
    try:
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            
            current_batch = i // batch_size + 1
            print(f"✅ Upserted batch {current_batch}/{total_batches}")
        
        print(f"🎉 Successfully upserted {len(points)} points to collection '{collection_name}'!")
        return len(points)
        
    except Exception as e:
        print(f"❌ Error during upsert: {e}")
        return 0

def search_similar_texts(qdrant_client, collection_name, query, api_key=None, top_k=3):
    """Search for similar texts in Qdrant with better error handling"""
    print("🔍 Searching for similar texts...")
    
    # First, verify the collection exists
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' found with {collection_info.points_count} points")
        
        if collection_info.points_count == 0:
            print(f"⚠️ Collection '{collection_name}' is empty")
            return None
            
    except Exception as collection_error:
        print(f"❌ Collection '{collection_name}' not found: {collection_error}")
        # List available collections for debugging
        try:
            collections = qdrant_client.get_collections()
            available_collections = [c.name for c in collections.collections]
            print(f"Available collections: {available_collections}")
        except Exception as list_error:
            print(f"Could not list collections: {list_error}")
        return None
    
    # Generate query embedding
    query_embeddings = generate_query_embedding(query, api_key)
    
    if not query_embeddings:
        print("❌ Failed to generate query embedding")
        return None
    
    try:
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embeddings,
            limit=top_k,
            with_payload=True
        )
        
        print(f"✅ Found {len(results.points)} results")
        return results
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return None


def get_robust_qdrant_client(storage_path="./qdrant_storage"):
    """
    Get a Qdrant client with multiple fallback strategies
    Updated to be more consistent
    """
    # Ensure absolute path consistency
    storage_path = os.path.abspath(storage_path)
    
    # Strategy 1: Try normal disk storage
    try:
        return setup_local_qdrant_storage(storage_path)
    except Exception as e1:
        print(f"Strategy 1 failed: {e1}")
        
        # Strategy 2: Try with unique storage path only if it's a lock issue
        if "already accessed by another instance" in str(e1):
            try:
                unique_path = f"{storage_path}_{int(time.time())}"
                print(f"Trying unique path: {unique_path}")
                return setup_local_qdrant_storage(unique_path)
            except Exception as e2:
                print(f"Strategy 2 failed: {e2}")
        
        # Strategy 3: Use in-memory (fallback)
        try:
            return
        except Exception as e3:
            print(f"All strategies failed. Last error: {e3}")
            return None

def initialize_database(pdf_path, collection_name="embeddings_collection", api_key=None, storage_path="./qdrant_storage", force_reprocess=False):
    """Initialize the database with PDF content using local disk storage"""
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")

    print("🚀 Initializing Qdrant database with local disk storage...")
    
    # Initialize Qdrant client with disk storage
    qdrant_client = setup_local_qdrant_storage(storage_path)
    if not qdrant_client:
        raise ValueError("Failed to initialize Qdrant client")

    # Check if collection exists and has data
    has_data, collection_info = check_collection_exists_and_has_data(qdrant_client, collection_name)
    
    if has_data and not force_reprocess:
        print("⚡ Collection already populated. Skipping PDF processing and embedding generation.")
        print("💡 Use force_reprocess=True if you want to reprocess the data.")
        return qdrant_client, collection_name
    
    # Only process if collection doesn't have data or force_reprocess is True
    print("📄 Loading and splitting PDF...")
    texts = load_and_split_pdf(pdf_path)
    print(f"✅ Loaded {len(texts)} text chunks")

    print("🔢 Generating embeddings...")
    raw_texts = [doc.page_content for doc in texts]
    embeddings = get_embeddings(raw_texts, api_key)

    if embeddings and len(embeddings) > 0:
        vector_size = len(embeddings[0])
        print(f"✅ Generated embeddings with dimension: {vector_size}")
    else:
        raise ValueError("Failed to generate embeddings")

    # Initialize collection if it doesn't exist
    if not collection_info:
        init_qdrant_collection(qdrant_client, collection_name, vector_size)

    print("💾 Uploading embeddings to Qdrant disk storage...")
    upserted_count = upsert_embeddings_to_qdrant(qdrant_client, collection_name, embeddings, texts)
    
    if upserted_count > 0:
        print("🎉 Database initialization complete!")
        print(f"📁 Data stored at: {os.path.abspath(storage_path)}")
    else:
        print("⚠️ Database initialization completed with errors.")
    
    return qdrant_client, collection_name

def get_collection_info(qdrant_client, collection_name):
    """Get information about a collection"""
    try:
        info = qdrant_client.get_collection(collection_name)
        print(f"\n📊 Collection '{collection_name}' Info:")
        print(f"   - Points count: {info.points_count}")
        print(f"   - Vector size: {info.config.params.vectors.size}")
        print(f"   - Distance metric: {info.config.params.vectors.distance}")
        return info
    except Exception as e:
        print(f"❌ Error getting collection info: {e}")
        return None

def delete_collection(qdrant_client, collection_name):
    """Delete a collection"""
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"🗑️ Collection '{collection_name}' deleted successfully")
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")

if __name__ == "__main__":
    pdf_path = "./Data/GardenSuits.pdf"
    storage_path = "./my_qdrant_data" 
    
    try:
        # Normal initialization - will skip processing if data exists
        client, collection = initialize_database(
            pdf_path=pdf_path,
            collection_name="my_documents",
            storage_path=storage_path
        )
        
        # If you want to force reprocessing even if data exists, use:
        # client, collection = initialize_database(
        #     pdf_path=pdf_path,
        #     collection_name="my_documents",
        #     storage_path=storage_path,
        #     force_reprocess=True
        # )
        
        # Example search
        query = "Garden Suits?"
        results = search_similar_texts(client, collection, query)
        
        if results:
            print("\n🔍 Search Results:")
            for i, point in enumerate(results.points):
                print(f"\n{i+1}. Score: {point.score:.4f}")
                print(f"   Text: {point.payload['text'][:200]}...")
                print(f"   Hierarchy: {point.payload['hierarchy']}")
                
    except Exception as e:
        print(f"❌ Error: {e}")