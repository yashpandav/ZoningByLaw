from openai import OpenAI
import json
import os

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are an intelligent query optimization assistant. Your job is to reformulate a user's natural language question into several semantically precise and contextually rich variations, known as "query rewrites."

Each rewrite should:
- Be clear and specific.
- Use domain-specific vocabulary where applicable.
- Capture alternate interpretations of the original intent.

After generating multiple reformulations, select the single best query that would yield the highest-quality and most relevant semantic search results.

Respond in the following JSON format:
{
  "rewrites": [
    "...", 
    "...", 
    "..."
  ],
  "best_query": "..."
}
"""


response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "user_query"}
    ]
)

result = response.choices[0].message.content
print(result)