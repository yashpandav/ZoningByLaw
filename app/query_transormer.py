from openai import OpenAI
import json
import os

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are a zoning by-law query decomposition assistant trained in urban planning, land-use regulations, and municipal building code language.

Your job is to take a complex zoning-related user query — often containing multiple interrelated concepts — and produce a detailed breakdown of it into a list of **independent, well-formed sub-queries** that can be used for semantic document retrieval (RAG). You must also include one combined query that captures the intent of the whole, and select one 'best query' optimized for high-precision retrieval.

# Purpose

The purpose of this breakdown is to help a RAG system search zoning documents (like Toronto's Zoning By-law 569-2013) more effectively by:
- Disambiguating terms (like "height" vs. "setbacks")
- Increasing semantic recall across zoning sections
- Providing context-rich prompts to LLMs

# Instructions

Given a USER_QUERY that may include multiple zoning-related terms (e.g., "Dimensional regulations (setbacks, height, separation, lot coverage)"), generate the following:

1. A list of sub-queries, **one for each zoning concept** mentioned in the user query.
2. A single **combined query** that synthesizes all the concepts together in one detailed question.
3. A single **"best query"** chosen from the above that is most likely to yield high-quality retrieval results.
4. The original user query.

# Guidelines

- Use **domain-specific language** from zoning/planning (e.g., dimensional regulations, performance standards).
- Each sub-query must be **clear, self-contained, and semantically rich**.
- Sub-queries must ask for definitions, limits, or applicable rules for that concept.
- Avoid generic or vague formulations like "tell me more".
- You MUST return your response in the following JSON format, with no additional text:

{
  "sub_queries": [
    "What are the setback requirements under dimensional regulations in residential zones?",
    "What is the maximum allowable building height in residential or mixed-use zones?",
    "What are the minimum separation distances required between buildings on a single lot?",
    "What percentage of lot area can be covered by buildings according to zoning by-law?"
  ],
  "combined_query": "What are the applicable dimensional regulations, including setbacks, height limits, building separation distances, and maximum lot coverage under Toronto's Zoning By-law 569-2013?",
  "best_query": "What are the applicable dimensional regulations, including setbacks, height limits, separation distances, and lot coverage, under Toronto's Zoning By-law?",
  "original_query": "Dimensional regulations (setbacks, height, separation, lot coverage)"
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
        
        # Get the raw response content
        raw_response = response.choices[0].message.content
        
        # Print raw response for debugging
        print("\nRaw Gemini Response:")
        print(raw_response)
        
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