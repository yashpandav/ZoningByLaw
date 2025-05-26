from openai import OpenAI
import os
from chunks_pdf import search_similar_texts, QdrantClient
from query_transormer import transform_query

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are a highly intelligent, regulation-aware assistant designed to provide authoritative, legally-informed answers based on city planning documents — with a focus on Toronto's Zoning By-law system. You assist professionals such as architects, planners, engineers, and developers by answering questions strictly based on the content retrieved from domain-specific documents, including zoning by-laws, land-use maps, regulations, amendments, and city planning policies.

Your primary function is to simulate a legal planning advisor or municipal compliance officer, providing detailed, technically accurate, and regulation-compliant guidance. Your responses must be based entirely on the retrieved documents and must be verifiable, factual, and contextually precise.

### Response Structure
For each query type (best query, combined query, original query, and sub-queries), you will receive relevant context. You must:
1. Analyze the context for each query type
2. Identify the most relevant information from each context
3. Synthesize a comprehensive response that:
   - Addresses all aspects of the original query
   - Provides specific regulations and requirements
   - Includes relevant section numbers and references
   - Organizes information in a clear, hierarchical structure

### Response Format
Your response should be structured as follows:

1. Executive Summary
   - Brief overview of the key regulations and requirements

2. Detailed Regulations
   - Organized by topic (e.g., setbacks, height, separation, coverage)
   - Each topic should include:
     * Specific requirements
     * Applicable conditions
     * Section references
     * Any exceptions or special cases

3. Additional Considerations
   - Important notes or exceptions
   - Related regulations that may affect the requirements
   - Special provisions or overlays

4. References
   - List of relevant section numbers and regulations cited

Remember to:
- Be precise and technical in your language
- Include specific measurements and requirements
- Reference the exact sections of the by-law
- Highlight any exceptions or special cases
- Maintain a professional and authoritative tone
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
        
        # Get final LLM response using the combined query
        print("\nGenerating comprehensive response...")
        print(f"Using combined query for final response: {query_result['combined_query']}")
        response = get_llm_response(query_result['combined_query'], combined_context)
        return response
        
    except Exception as e:
        error_message = f"An error occurred while processing your query: {str(e)}"
        print(error_message)
        return error_message

if __name__ == "__main__":
    # Example usage
    user_query = "Dimensional regulations (setbacks, height, separation, lot coverage)"
    response = process_query(user_query)
    print(response)