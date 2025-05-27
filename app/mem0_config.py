from mem0 import Memory
from openai import OpenAI
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

print("Starting script...")

qdrant = QdrantClient(host="localhost", port=6333)

qdrant.recreate_collection(
    collection_name="test",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
QUADRANT_HOST = "localhost"

NEO4J_URL="neo4j+s://2090d8f9.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="lxQVdjyPs9Dybd6mnlZjCWVPiHur2tSBJzRymPfJFZ8"

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
        }
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-1.5-flash-latest",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URL, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
}

print("Initializing Memory with config...")
m = Memory.from_config(config_dict=config)
print("Memory initialized successfully")

