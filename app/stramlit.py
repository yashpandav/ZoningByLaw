import streamlit as st
import os
from openai import OpenAI
from chunks_pdf import search_similar_texts, QdrantClient
from langsmith import wrappers
import time

# Page configuration
st.set_page_config(
    page_title="Toronto Zoning By-law Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
@st.cache_resource
def initialize_openai_client():
    return wrappers.wrap_openai(OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    ))

# Initialize Qdrant client
@st.cache_resource
def initialize_qdrant():
    return QdrantClient(host="localhost", port=6333)

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

def get_llm_response(user_query, context, client):
    """Get response from LLM using the user query and retrieved context"""
    # You'll need to define SYSTEM_PROMPT or pass it as a parameter
    SYSTEM_PROMPT = """You are an expert assistant for Toronto's Zoning By-law regulations. 
    Provide clear, accurate, and comprehensive responses based on the provided context. 
    Structure your answers with relevant sections, codes, and specific requirements."""
    
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
                "role": "user", "content": formatted_query}
        ]
    )
    return response.choices[0].message.content

def process_query(user_query, client, qdrant):
    """Process a user query by retrieving context and getting LLM response"""
    COLLECTION_NAME = "jina_embeddings_collection2"
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    
    try:
        # Check if collection exists
        try:
            qdrant.get_collection(COLLECTION_NAME)
        except Exception:
            # If collection doesn't exist, initialize the database
            st.warning("Collection not found. Initializing database...")
            with st.spinner("Setting up database... This may take a few minutes."):
                from chunks_pdf import initialize_database
                PDF_PATH = "./Data/GardenSuits.pdf" 
                initialize_database(PDF_PATH, COLLECTION_NAME)
            st.success("Database initialized successfully!")
        
        # Search for relevant information
        with st.spinner("Searching for relevant information..."):
            results = search_similar_texts(qdrant, COLLECTION_NAME, user_query, JINA_API_KEY)
            context = format_search_results(results, user_query)
        
        # Get final LLM response
        with st.spinner("Generating comprehensive response..."):
            response = get_llm_response(user_query, context, client)
        
        return response, results
        
    except Exception as e:
        error_message = f"An error occurred while processing your query: {str(e)}"
        st.error(error_message)
        return error_message, None

def main():
    # Header
    st.title("🏢 Toronto Zoning By-law Assistant")
    st.markdown("Ask questions about Toronto's zoning regulations and get detailed, contextual answers.")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This assistant helps you navigate Toronto's Zoning By-law documents by:
        - 🔍 Searching through relevant sections
        - 📖 Providing contextual answers
        - 🎯 Highlighting specific regulations
        
        **How to use:**
        1. Enter your question in the text area
        2. Click 'Search & Analyze'
        3. Review the comprehensive response
        """)
        
        st.header("⚙️ Settings")
        show_sources = st.checkbox("Show source results", value=True)
        max_results = st.slider("Max search results", min_value=3, max_value=10, value=5)
    
    # Initialize clients
    try:
        client = initialize_openai_client()
        qdrant = initialize_qdrant()
    except Exception as e:
        st.error(f"Failed to initialize clients: {str(e)}")
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Question")
        
        # Query input
        user_query = st.text_area(
            "Enter your question about Toronto's zoning by-law:",
            placeholder="e.g., What are the requirements for garden suites?",
            height=100
        )
        
        # Example queries
        st.markdown("**Example questions:**")
        example_queries = [
            "What are the setback requirements for residential buildings?",
            "How do I apply for a garden suite permit?",
            "What are the height restrictions in residential zones?",
            "What parking requirements apply to multi-unit buildings?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"📝 {example}", key=f"example_{i}"):
                    user_query = example
                    st.rerun()
    
    with col2:
        st.header("🚀 Actions")
        search_button = st.button(
            "🔍 Search & Analyze", 
            type="primary",
            use_container_width=True,
            disabled=not user_query.strip()
        )
        
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Process query
    if search_button and user_query.strip():
        st.markdown("---")
        
        # Show the query being processed
        st.subheader("🔄 Processing Query")
        st.info(f"**Query:** {user_query}")
        
        # Process the query
        start_time = time.time()
        response, results = process_query(user_query, client, qdrant)
        processing_time = time.time() - start_time
        
        if results is not None:
            # Display response
            st.subheader("✅ Response")
            st.markdown(response)
            
            # Show processing time
            st.caption(f"⏱️ Processed in {processing_time:.2f} seconds")
            
            # Show source results if enabled
            if show_sources and results:
                st.markdown("---")
                st.subheader("📚 Source Results")
                
                for i, res in enumerate(results.points[:max_results], start=1):
                    with st.expander(f"Source #{i} (Relevance: {res.score:.3f})"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if res.payload.get('hierarchy'):
                                st.markdown(f"**Section:** {res.payload.get('hierarchy')}")
                            if res.payload.get('heading_code'):
                                st.markdown(f"**Code:** {res.payload.get('heading_code')}")
                            if res.payload.get('heading_title'):
                                st.markdown(f"**Title:** {res.payload.get('heading_title')}")
                        
                        with col2:
                            st.markdown(f"**Content:**")
                            st.markdown(res.payload.get('text', 'No text found'))
    
    # Chat history (optional enhancement)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if user_query and search_button:
        st.session_state.chat_history.append({
            'query': user_query,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔧 **Note:** Make sure your environment variables (`GOOGLE_API_KEY`, `JINA_API_KEY`) are properly set "
        "and Qdrant is running on localhost:6333"
    )

if __name__ == "__main__":
    main()