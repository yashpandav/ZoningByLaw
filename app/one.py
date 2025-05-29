import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_query_embedding(query):
        print("Processing query...")
        query = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
        )
        return query["embedding"]