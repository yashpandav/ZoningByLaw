import requests
import json
from qdrant_client import QdrantClient
import os
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path, chunk_size=800, chunk_overlap=200):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text = splitter.split_documents(documents=documents)
    print(text)
    return text

def get_jina_embeddings(texts, jina_api_key, model="jina-embeddings-v3", task="text-matching"):
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
    # Check if collection exists; if not, create it
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
    points = []
    for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
        # If text is a Document, get its content, else use as is
        content = text.page_content if hasattr(text, "page_content") else text
        points.append({
            "id": idx,
            "vector": embedding,
            "payload": {"text": content}
        })
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Upserted {len(points)} points to collection '{collection_name}'.")


def search_similar_texts(qdrant_client, collection_name, query, jina_api_key, top_k=3):
    query_embedding = get_jina_embeddings([query], jina_api_key)[0]

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k
    )
    return results

def main():
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    PDF_PATH = "../GardenSuits.pdf"
    COLLECTION_NAME = "jina_embeddings_collection2"

    # Step 1: Load and split PDF
    print("Loading and splitting PDF...")
    texts = load_and_split_pdf(PDF_PATH)

    # Step 2: Get embeddings for the texts
    print("Generating embeddings...")
    raw_texts = [doc.page_content for doc in texts]
    embeddings = get_jina_embeddings(raw_texts, JINA_API_KEY)
    vector_size = len(embeddings[0])

    # Step 3: Connect to Qdrant and create collection if needed
    qdrant = QdrantClient(host="localhost", port=6333)
    init_qdrant_collection(qdrant, COLLECTION_NAME, vector_size)

    # Step 4: Upload embeddings and texts to Qdrant
    print("Uploading embeddings to Qdrant...")
    upsert_embeddings_to_qdrant(qdrant, COLLECTION_NAME, embeddings, texts)

    # Step 5: Query Qdrant with a sample search
    query = "Minimum Separation between a Residential Building"
    print(f"Searching for top results matching query: {query}")
    results = search_similar_texts(qdrant, COLLECTION_NAME, query, JINA_API_KEY)

    for i, res in enumerate(results.points, start=1):
        print(f"Result #{i}")
        print(f"Score: {res.score:.4f}")
        print(f"Text: {res.payload.get('text', 'No text found')}")
        print("------")

if __name__ == "__main__":
    main()
