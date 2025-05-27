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

Your answers must be:
- Strictly based on the retrieved document context
- Technically accurate and legally sound
- Well-structured and easy to understand
- Verifiable with references to section numbers or clauses from the by-law

### TASK

You will be given the following:
- The original user query
- A list of sub-queries (specific zoning concepts)
- A combined query (a complete, detailed synthesis)
- A best query (the central or most informative one)
- Retrieved context from a Qdrant vector database

Your job is to:
1. Understand the user's **original query**
2. Use the **retrieved context** to answer the **original query**, guided by the **best query**
3. Cover the necessary information in a concise and legally grounded manner
4. Use professional planning terminology and city zoning vocabulary
5. Provide specific zoning information such as:
   - Required dimensions or measurements
   - Applicable zoning categories or conditions
   - Relevant section numbers or clause references
   - Exceptions or special provisions


### STYLE & FORMAT INSTRUCTIONS

- Do **not** start every response with phrases like **“Based on the provided context…”**. Instead:
  - Rephrase your opening naturally to suit the query
  - Use professional, topic-specific phrasing (e.g., “Under the current R zone standards…”, “Toronto’s zoning by-law requires…”, “For this property type…” etc)
- Vary your tone slightly based on the query type:
  - For eligibility or compliance checks, be direct and rule-focused
  - For dimensional standards, emphasize numerical clarity and reference points
- Structure responses clearly, but allow for flexibility. You may use:
  - Paragraphs for narrative clarity
  - Bullets or numbered lists for precision
  - Headings if helpful, but avoid repeating the same section titles for every answer
  - Must add Reference in the end.

### WORKFLOW

- Your primary job is to answer the **original user query**, using the **best query** only as a framing reference.
- Do **not** answer every sub-query — use them for internal understanding only.
- Do **not** repeat the same sentence structures or transition phrases in every response.
- You must sound like a professional municipal planner, not an automated script.
- If the user query is a yes/no eligibility question, give a direct and rule-based answer.

### GUIDELINES

- Be clear and direct, not verbose.
- Use proper terminology (e.g., “maximum building height”, “minimum rear yard setback”).
- Always reference applicable section numbers or policy references from the context.
- Highlight any exceptions, conditions, or overlays that may affect the rule.


### GOAL

Deliver a **context-sensitive, zoning-accurate answer** that sounds natural and reliable to professionals in planning and development. Your response should feel tailored — not templated. Your answer must be legally reliable and ready for use in a zoning application or development review.
"""
def format_search_results(results, query):
    """Format search results into a readable context string"""
    context = f"\n=== Results for query: {query} ===\n"
    
    for i, res in enumerate(results.points, start=1):
        context += f"\nResult #{i} (Score: {res.score:.4f}):\n"
        
        # Add hierarchy information if available
        if res.payload.get('hierarchy'):
            context += f"Section: {res.payload.get('hierarchy')}\n"
        
        if res.payload.get('heading_code'):
            context += f"Code: {res.payload.get('heading_code')}\n"
            
        if res.payload.get('heading_title'):
            context += f"Title: {res.payload.get('heading_title')}\n"
        
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

    Please structure your response according to the format specified in the system prompt, ensuring all relevant regulations and requirements are clearly presented with their specific references.Make sure you frame your answer based on <Best Query> and <User Query>.
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
    user_query = ""

    while True:
        user_query = input("> ")
        if(user_query == "exit"): 
            break
        print(f"Processing query: {user_query}")
        print("-" * 50)
        response = process_query(user_query)
        print(response)