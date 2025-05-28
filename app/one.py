from google import genai
import os

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_query_embedding(query):
        query = client.models.embed_content(
                model="text-embedding-004",
                contents=query,
                task_type="retrieval_query"
        )
        return query.embeddings[0].values