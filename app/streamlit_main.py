import streamlit as st
from openai import OpenAI
import os
from chunks_pdf import search_similar_texts, QdrantClient, initialize_database
from langsmith import wrappers

# Configure page settings
st.set_page_config(
    page_title="Toronto Zoning Assistant",
    page_icon="🏙️",
    layout="wide"
)

client = wrappers.wrap_openai(OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
))

SYSTEM_PROMPT = """
You are a highly skilled planning and zoning assistant specialized in interpreting Toronto's Zoning By-law. You assist professionals such as architects, engineers, urban planners, and developers by providing precise, legally compliant answers based solely on official zoning documents.

These include:
- Zoning By-law 569-2013
- Zoning amendments and overlays
- Land-use maps and municipal planning policies

Your job is to interpret the **retrieved context** and produce a clear, reliable, and regulation-based answer to the user's question. Your answers must be practical, professional, and ready for use in a zoning application or review.

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

You will receive:
- A **user query**
- A **set of document chunks** retrieved from a vector database (Qdrant)

You must:
1. Analyze the user's question and intent
2. Use the retrieved zoning content to construct a legally grounded, complete response
3. Include precise zoning rules, standards, and exceptions — **only if present in the retrieved context**
4. Provide the following where relevant:
   - Dimensional standards (e.g., height, setbacks, lot coverage)
   - Use permissions and zoning conditions
   - Special provisions or site-specific exceptions
   - Section or clause references

### STYLE & FORMAT INSTRUCTIONS
Your answer must be:
- Clear, professional, and tailored to the user's query
- Structurally flexible: you may use paragraphs, bullet points, or headings
- **Dynamic in tone**:
  - Be direct and firm for yes/no eligibility or compliance queries
  - Use numerical clarity for dimensional or spatial rules
  - Rephrase the opening sentence to match the query — never use: _"Based on the provided context..."_

#### Always end with:
**References** — include section numbers, clauses, or by-law identifiers when stated in the context

### GUIDELINES
- Use only the content retrieved; **do not guess or fabricate**
- Reflect the exact terminology and rules from the zoning documents
- If a regulation depends on conditions (e.g., setbacks by lot width), mention that clearly
- If the answer is not found in the context, say:
  > "The retrieved document does not contain this information."
- Be clear and direct, not verbose.

### GOAL

Produce a well-organized, legally trustworthy answer that aligns with Toronto's zoning framework. Your response must be actionable and appropriate for submission in design reviews, rezoning applications, or planning consultations.
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
            st.warning("Collection not found. Initializing database...")
            PDF_PATH = "./Data/GardenSuits.pdf" 
            initialize_database(PDF_PATH, COLLECTION_NAME)
            st.success("Database initialized successfully!")
        
        # Search for relevant information
        with st.spinner("Searching for relevant information..."):
            results = search_similar_texts(qdrant, COLLECTION_NAME, user_query, JINA_API_KEY)
            context = format_search_results(results, user_query)
        
        # Get final LLM response
        with st.spinner("Generating comprehensive response..."):
            response = get_llm_response(user_query, context)
            return response, context
        
    except Exception as e:
        error_message = f"An error occurred while processing your query: {str(e)}"
        st.error(error_message)
        return error_message, ""

# Streamlit UI
st.title("Toronto Zoning By-law Assistant")
st.markdown("""
### Ask questions about Toronto's zoning regulations

This tool helps professionals like architects, engineers, urban planners, and developers 
get precise, legally compliant answers based on Toronto's official zoning documents.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses AI to search through Toronto's zoning documents and provide 
    accurate, regulation-based responses to your queries.
    
    **Examples of questions you can ask:**
    - What are the setback requirements for residential zones?
    - What is the maximum height allowed for a detached house?
    - What are the requirements for garden suites?
    - Can I build a secondary suite in my basement?
    """)
    
    # PDF selection (could be expanded in the future)
    st.subheader("Document Settings")
    pdf_options = ["GardenSuits.pdf", "Zoning-By-law-Part-1.pdf", "Zoning-By-law-Part-2.pdf", "Zoning-By-law-Part-3.pdf"]
    selected_pdf = st.selectbox("Select zoning document", pdf_options, index=0, disabled=True)
    st.caption("Currently using: GardenSuits.pdf")
    
    # Advanced settings
    st.subheader("Advanced Settings")
    show_context = st.checkbox("Show search context", value=False)

# Main query interface
user_query = st.text_area("Enter your zoning question:", height=100, placeholder="e.g., What are the requirements for garden suites in Toronto?")

if st.button("Submit Question", type="primary"):
    if user_query:
        response, context = process_query(user_query)
        
        # Display the response
        st.markdown("### Answer")
        st.markdown(response)
        
        # Optionally show the context
        if show_context:
            with st.expander("View search context"):
                st.text(context)
    else:
        st.warning("Please enter a question to get started.")

if __name__ == "__main__":
    user_query = ""

    while True:
        user_query = input("> ")
        if(user_query == "exit"): 
            break
        print(f"Processing query: {user_query}")
        print("-" * 50)
        response, _ = process_query(user_query)
        print(response)