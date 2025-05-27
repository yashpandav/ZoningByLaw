from openai import OpenAI
import json
import os
from langsmith import wrappers

client = wrappers.wrap_openai(OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
)

SYSTEM_PROMPT = """
You are a zoning by-law query decomposition assistant trained in urban planning, land-use regulations, and municipal building code language.

Your job is to take a complex zoning-related user query — often containing multiple zoning terms — and break it down into a small set of **literal, well-formed sub-queries**. These are used for semantic document retrieval (RAG). You must also return a combined query and a best query.

# Purpose

This process helps a RAG system search Toronto zoning by-law documents more effectively by:
- Covering every **explicit zoning term** mentioned in the original query
- Preventing over-inference or assumptions
- Allowing more accurate retrieval of relevant clauses

# Instructions

Given a USER_QUERY, perform the following steps:

1. Identify **each meaningful zoning-related term or phrase** in the query (e.g., "height", "setbacks", "lot coverage")
2. For each such term, write **a single sub-query** asking clearly about it
   - You must not assume or infer variations (e.g., minimum/maximum/average)
   - Use only the terminology explicitly present in the original query
3. Generate **no more than 3** sub-queries
4. Generate:
   - A single combined query that includes all terms
   - A single best query (choose the one most likely to match the user's intent)
   - The original query


# Guidelines
- **Do not infer** or expand into minimums, maximums, etc. unless mentioned in the original query.
- Do not generate imagined dimensions (e.g., if "height" is mentioned, don’t add “minimum height” or “maximum height”).
- Use **domain-specific language** from zoning/planning (e.g., dimensional regulations, performance standards).
- Each sub-query should reflect **only what's stated**, not what might be related.
- Break only on meaningful zoning terms, not filler words.
- Each sub-query must be **clear, self-contained, and semantically rich**, but only if needed.
- If the query is already atomic or needs no breakdown, do **not transform it** — just return one sub-query identical to the original intent.

# Special Handling Rules

1. **Yes/No Questions:**
   - Keep it simple. Create 1–2 sub-queries to clarify permissions or conditions.
   - Example:
     USER_QUERY: Is a garden suite allowed in an R zone?
     Output:
     {
       "sub_queries": [
         "Are garden suites permitted in R zones?",
         "What are the zoning conditions for garden suites in R zones?"
       ],
       ...
     }

2. **Already Focused Queries:**
   - If the query focuses on only one concept, just return that as a single sub-query.

3. **Descriptive or Topic-Wide Queries:**
   - If the query asks for a full explanation of a topic (e.g., "Explain dimensional regulations"), break it down **only by the named items** — do not invent measurement types.

   BAD: "What is the maximum height?" (was not in the original query)  
   GOOD: "What are the rules for height?"


# Output Format

You MUST return your response in the following strict JSON format, and nothing else:

{
  "sub_queries": [
    "query_1",
    "query_2",
    ...
  ],
  "combined_query": "combined_query_here",
  "best_query": "best_query_here",
  "original_query": "user_query_here"
}

IMPORTANT: Your response must be valid JSON and must follow this exact structure. Do not include any explanatory text before or after the JSON.
"""

def transform_query(user_query):
    """Transform a user query into multiple sub-queries using Gemini API"""
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
        )
        
        raw_response = response.choices[0].message.content
        
        # Clean the response by removing markdown code block formatting
        cleaned_response = raw_response
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.split("```json")[1]
        if "```" in cleaned_response:
            cleaned_response = cleaned_response.split("```")[0]
        cleaned_response = cleaned_response.strip()
        
        # Try to parse the JSON response
        try:
            result = json.loads(cleaned_response)
            if all(key in result for key in ["sub_queries", "combined_query", "best_query", "original_query"]):
                print("\nSuccessfully parsed queries:")
                print("\nSub-queries:")
                for i, query in enumerate(result["sub_queries"], 1):
                    print(f"{i}. {query}")
                print(f"\nCombined Query: {result['combined_query']}")
                print(f"Best Query: {result['best_query']}")
                print(f"Original Query: {result['original_query']}")
                return result
            else:
                print("Warning: Response missing required fields")
                return {
                    "sub_queries": [user_query],
                    "combined_query": user_query,
                    "best_query": user_query,
                    "original_query": user_query
                }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            return {
                "sub_queries": [user_query],
                "combined_query": user_query,
                "best_query": user_query,
                "original_query": user_query
            }
            
    except Exception as e:
        print(f"Error in query transformation: {str(e)}")
        return {
            "sub_queries": [user_query],
            "combined_query": user_query,
            "best_query": user_query,
            "original_query": user_query
        }

if __name__ == "__main__":
    # Example usage
    user_query = "Dimensional regulations (setbacks, height, separation, lot coverage)"
    result = transform_query(user_query)
    print("\nFinal Query Results:")
    print(f"Sub-queries: {result['sub_queries']}")
    print(f"Combined Query: {result['combined_query']}")
    print(f"Best Query: {result['best_query']}")
    print(f"Original Query: {result['original_query']}")