from openai import OpenAI
import os
from chunks_pdf import search_similar_texts, QdrantClient
from query_transormer import transform_query
from langsmith import wrappers

client = wrappers.wrap_openai(OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
))

SYSTEM_PROMPT = """
You are a knowledgeable and regulation-aware assistant that provides clear, professional, and legally accurate answers based on planning and zoning documents — especially focused on the Toronto Zoning By-law.

You help users like architects, engineers, planners, and developers by giving reliable, regulation-based responses using the context retrieved from zoning documents such as:
- City zoning by-laws
- Land-use maps
- Municipal planning policies
- Zoning amendments and overlays
- and more problems...

Your answers must be:
- Strictly based on the retrieved document context
- Technically accurate and legally sound
- Well-structured and easy to understand
- Verifiable with references to section numbers or clauses from the by-law

---

### TASK

You will be given the following:
- The original user query
- A list of sub-queries (specific concepts from the original question)
- A combined version of the query
- A best query (most important or central one)
- Retrieved context from a Qdrant vector database

Your job is to:
1. Understand the user's **original question**
2. Use the **retrieved context** to answer the query as thoroughly and accurately as possible
3. Cover all topics from the sub-queries, best query, and combined query
4. Use planning terminology and city zoning vocabulary
5. Provide specific details like:
   - Required measurements
   - Zoning categories or conditions
   - Section numbers and references
   - Exceptions or special rules

---

### RESPONSE FORMAT

Organize your response like this (or in another clearly structured way):

1. **Executive Summary**  
   - A short overview of the key zoning rules relevant to the query

2. **Detailed Regulations**  
   - Break this into sections by topic (e.g., setbacks, height, lot coverage)
   - For each section, include:
     * The rule or regulation
     * Specific numbers or limits
     * Applicable zones or conditions
     * Section references from the zoning by-law
     * Exceptions or overlays if any

3. **Additional Considerations**  
   - Other regulations that could impact the topic (e.g., heritage overlays, minor variances)

4. **Optional: Other Notes**  
   - Any extra information that could be useful but wasn’t directly asked

5. **References**  
   - A list of section numbers and zoning document references used in your answer

---

### WORKFLOW

- You will receive:  
  **Original User Query:** <USER_QUERY> 
  **Sub Queries:** <SUB_QUERIES>  
  **Best Query:** <BEST_QUERY>  
  **Combined Query:** <COMBINED_QUERY>  

- First, read the **original query** to understand the user's intent
- Then, answer the query based on all the provided context and queries
- If the query implies a yes/no answer (e.g., “Is X allowed?”), be direct — but also explain your answer with supporting detail

---

### GUIDELINES

- Stay factual — only use the given context
- Be formal, clear, and professional
- Use specific planning terms (e.g., “minimum setback distance” instead of “space between buildings”)
- Always include section numbers if mentioned in context
- Highlight exceptions or special provisions where applicable

Your response must be accurate, complete, and trustworthy — like that of a city planning officer or legal advisor.
"""

def format_search_results(results, query):
    """Format search results into a readable context string"""
    context = f"\n=== Results for query: {query} ===\n"
    for i, res in enumerate(results.points, start=1):
        context += f"\nResult #{i}:\n"
        context += f"Text: {res.payload.get('text', 'No text found')}\n"
        context += "---\n"
    return context

def get_llm_response(user_query, context):
    """Get response from LLM using the user query and retrieved context"""
    formatted_query = f"""
    Please provide a comprehensive response to the following query based on the provided context from Toronto's Zoning By-law:

    QUERY: {user_query}

    CONTEXT:
    {context}

    Please structure your response according to the format specified in the system prompt, ensuring all relevant regulations and requirements are clearly presented with their specific references.
    """
    
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[  
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": formatted_query
            }
        ]
    )
    return response.choices[0].message.content

def process_query(user_query):
    """Process a user query by retrieving context and getting LLM response"""
    # Initialize Qdrant client
    qdrant = QdrantClient(host="localhost", port=6333)
    COLLECTION_NAME = "jina_embeddings_collection2"
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    
    try:
        # Check if collection exists
        try:
            qdrant.get_collection(COLLECTION_NAME)
        except Exception:
            # If collection doesn't exist, initialize the database
            print("Collection not found. Initializing database...")
            from chunks_pdf import initialize_database
            PDF_PATH = "../GardenSuits.pdf" 
            initialize_database(PDF_PATH, COLLECTION_NAME)
            print("Database initialized successfully!")
        
        # Transform the query into sub-queries
        print("\nTransforming query into sub-queries...")
        query_result = transform_query(user_query)
        
        if len(query_result["sub_queries"]) == 1 and query_result["sub_queries"][0] == user_query:
            print("Warning: Query transformation failed, using original query")
        else:
            print(f"\nGenerated {len(query_result['sub_queries'])} sub-queries:")
            for i, q in enumerate(query_result["sub_queries"], 1):
                print(f"{i}. {q}")
            print(f"\nCombined Query: {query_result['combined_query']}")
            print(f"Best Query: {query_result['best_query']}")
            print(f"Original Query: {query_result['original_query']}")
        
        # Collect results from all queries
        print("\nSearching for relevant information...")
        all_contexts = []
        
        # Search with each query type in order of specificity
        queries_to_process = [
            ("Best Query", query_result['best_query']),
            ("Combined Query", query_result['combined_query']),
            ("Original Query", query_result['original_query'])
        ]
        
        # Process main queries first
        for query_type, query in queries_to_process:
            print(f"\nProcessing {query_type}: {query}")
            results = search_similar_texts(qdrant, COLLECTION_NAME, query, JINA_API_KEY)
            context = format_search_results(results, query)
            all_contexts.append(context)
            
        
        # Then process sub-queries for detailed information
        print("\nProcessing sub-queries for detailed information:")
        for sub_query in query_result["sub_queries"]:
            print(f"\nProcessing sub-query: {sub_query}")
            results = search_similar_texts(qdrant, COLLECTION_NAME, sub_query, JINA_API_KEY)
            context = format_search_results(results, sub_query)
            all_contexts.append(context)
        
        # Combine all contexts
        combined_context = "\n".join(all_contexts)

        print(f"ALL CONTEXT : {all_contexts}")
        print(f"Combined Context Length: {len(combined_context)} characters")
        
        # Format the combined queries string with proper separation
        combined_all_queries = f"""
        Original User Query: {query_result['original_query']}

        Sub Queries: 
        {"\n".join([f"- {q}" for q in query_result["sub_queries"]])}

        Best Query: {query_result['best_query']}

        Combined Query: {query_result['combined_query']}
        """

        # Get final LLM response using the combined query
        print("\nGenerating comprehensive response...")
        print(f"All Queries Combined: \n{combined_all_queries}")

        response = get_llm_response(combined_all_queries, combined_context)
        return response

    except Exception as e:
        error_message = f"An error occurred while processing your query: {str(e)}"
        print(error_message)
        return error_message

if __name__ == "__main__":
    # Example usage
    user_query = "Parking and Bicycle Parking"
    response = process_query(user_query)
    print(response)