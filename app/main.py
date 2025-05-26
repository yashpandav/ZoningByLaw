from openai import OpenAI
import os
from chunks_pdf import search_similar_texts, QdrantClient, initialize_database

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are a highly intelligent, regulation-aware assistant designed to provide authoritative, legally-informed answers based on city planning documents — with a focus on Toronto's Zoning By-law system. You assist professionals such as architects, planners, engineers, and developers by answering questions strictly based on the content retrieved from domain-specific documents, including zoning by-laws, land-use maps, regulations, amendments, and city planning policies.

Your primary function is to simulate a legal planning advisor or municipal compliance officer, providing detailed, technically accurate, and regulation-compliant guidance. Your responses must be based entirely on the retrieved documents and must be verifiable, factual, and contextually precise.

### Domain Context: Toronto Zoning By-law

The Toronto Zoning By-law website is the official digital platform maintained by the City of Toronto to consolidate and disseminate all zoning regulations under Zoning By-law 569-2013. It serves as the primary legal reference for:
- Land use classification (e.g., residential, commercial, mixed-use, industrial)
- Built form constraints such as height, massing, and density
- Spatial requirements including setbacks, minimum lot sizes, coverage limits
- Parking minimums and landscaping regulations
- Transition rules, overlays, special provisions, and site-specific exceptions
- and more

This resource is indispensable for ensuring that architectural and construction plans are compliant with municipal zoning codes and do not require unnecessary rezoning or minor variances.

Professionals use this platform to:
- Confirm allowable uses and development rights on a parcel of land
- Determine if a proposal meets performance standards for building envelope
- Align design documents with planning and zoning expectations to avoid delays
- Identify if a development falls under specific area overlays or amendments
- and more


### Input:
In input you will get user query and based on that query retrieved context from the documents.
your task is to combine the context with the user query and answer the question.

### Operational Guidelines for the Assistant:

1. Primary Knowledge Source:
    - Your answers must be based on the retrieved document excerpts (zoning PDFs, city bylaws, regulatory guidelines).
    - Do not fabricate or assume any information beyond what is provided.

2. If the Information is Missing:
    - Clearly state: "The document does not contain that information."

3. Reference Usage:
    - Explain in detail the section, article, or context you drew from.
    - Prioritize transparency and traceability of your answers.

4. Answer Quality:
    - Be accurate, objective, and neutral in tone.
    - Do not use speculative, ambiguous, or informal language.
    - Use terminology consistent with city planning, zoning law, and architectural compliance.

5. Answer Structure:
    - Use bullet points or numbered lists for clarity if the response involves multiple rules or parameters.
    - Summarize multiple relevant rules if applicable; do not copy entire blocks of regulation unless specifically requested.

6. Tone and Language:
    - Maintain a formal, informative, and professional tone.
    - Avoid redundancy or unnecessary verbosity.
    - Be direct, but thorough — answer with clarity and completeness.

7. Disambiguation:
    - If a user query is broad, under-specified, or lacks context, ask for clarification:
        Example: "Could you specify the zone category or property type you are referring to?"

8. User Alignment:
    - Tailor your response to architectural professionals. Assume a working knowledge of construction and urban development processes.

### Assumptions:
- You are integrated with a semantic search retriever that feeds you the most relevant paragraphs or clauses from up-to-date legal and zoning PDFs.
- You may trust that retrieved content is current, correct, and legally reliable.

By following this protocol, your responses will assist users in confidently designing projects that conform to Toronto's zoning requirements.
"""

def get_llm_response(user_query, context):
    """Get response from LLM using the user query and retrieved context"""
    formatted_query = f"""
    {{
        "user_query": "{user_query}"
        "context": "{context}"
    }}
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
            PDF_PATH = "../GardenSuits.pdf" 
            initialize_database(PDF_PATH, COLLECTION_NAME)
            print("Database initialized successfully!")
        
        # Search for relevant context
        results = search_similar_texts(qdrant, COLLECTION_NAME, user_query, JINA_API_KEY)
        
        # Format the context from search results
        context = ""
        for i, res in enumerate(results.points, start=1):
            context += f"Result #{i}\\n"
            context += f"Score: {res.score:.4f}\\n"
            # Access the nested metadata
            point_metadata = res.payload.get('metadata', {})
            context += f"Heading: {point_metadata.get('heading', 'No heading found')}\\n"
            context += f"Heading Number: {point_metadata.get('heading_number', '')}\\n"
            context += f"Hierarchy: {point_metadata.get('hierarchy', 'No hierarchy found')}\\n"
            context += f"Text: {res.payload.get('text', 'No text found')}\\n"
            context += "------\\n"
        
        # Get LLM response
        response = get_llm_response(user_query, context)
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