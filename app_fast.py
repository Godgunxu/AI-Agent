import streamlit as st
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import time

# Configure Streamlit page
st.set_page_config(
    page_title="DocuMind AI - Intelligen            if st.button(sources_text, use_container_width=True): Document Assistant",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS function for ChatGPT-like UI with Dark/Light mode support
def get_css_styles(is_dark_mode=False):
    if is_dark_mode:
        # Dark mode colors
        bg_color = "#1a1a1a"
        text_color = "#e5e5e5"
        secondary_bg = "#2d2d2d"
        border_color = "#404040"
        user_msg_bg = "#0066cc"
        assistant_msg_bg = "#2d2d2d"
        header_bg = "#252525"
    else:
        # Light mode colors
        bg_color = "#ffffff"
        text_color = "#343541"
        secondary_bg = "#f7f7f8"
        border_color = "#e5e5e5"
        user_msg_bg = "#f7f7f8"
        assistant_msg_bg = "#ffffff"
        header_bg = "#ffffff"

    return f"""
<style>
    /* Hide Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Override Streamlit's default background */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Force text color for all elements */
    .stApp, .stApp *, .block-container, .block-container *, 
    .stMarkdown, .stMarkdown *, .stText, .stWrite, .stSelectbox,
    .element-container, .element-container *, .stColumn, .stColumn * {{
        color: {text_color} !important;
    }}
    
    /* Main container */
    .block-container {{
        max-width: 100%;
        padding: 1rem;
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Header */
    .main-header {{
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        color: {text_color};
        border-bottom: 1px solid {border_color};
        padding-bottom: 1rem;
        background-color: {header_bg};
    }}
    
    /* Chat messages */
    .chat-message {{
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 1rem;
        line-height: 1.5;
        color: {text_color};
    }}
    
    .user-message {{
        background-color: {user_msg_bg};
        color: {"#ffffff" if is_dark_mode else text_color};
        border: 1px solid {border_color};
    }}
    
    .assistant-message {{
        background-color: {assistant_msg_bg};
        color: {text_color};
        border: 1px solid {border_color};
    }}
    
    /* Status indicators */
    .status-good {{
        color: #10b981;
        font-weight: 500;
        padding: 0.5rem;
        border-radius: 4px;
        background-color: {secondary_bg};
        margin-bottom: 1rem;
    }}
    
    .status-info {{
        color: {text_color};
        font-size: 0.9rem;
        padding: 0.25rem 0.5rem;
        background-color: {secondary_bg};
        border-radius: 4px;
        border: 1px solid {border_color};
    }}
    
    /* Main content area with proper spacing for fixed input */
    .chat-container {{
        padding-bottom: 120px;
        max-width: 100%;
    }}
    
    /* Fixed chat input styling like ChatGPT */
    .stChatInput {{
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background-color: {bg_color} !important;
        border-top: 1px solid {border_color} !important;
        padding: 1rem !important;
        z-index: 999 !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1) !important;
    }}
    
    .stChatInput > div {{
        max-width: 768px !important;
        margin: 0 auto !important;
    }}
    
    /* Ensure body has proper padding for fixed input */
    .stApp {{
        padding-bottom: 100px !important;
    }}
    
    /* Source documentation styling */
    .source-docs {{
        background-color: {secondary_bg};
        border: 1px solid {border_color};
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: {text_color};
    }}
    
    /* Sample question buttons */
    div[data-testid="column"] button {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        text-align: left !important;
        width: 100% !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
    }}
    
    div[data-testid="column"] button:hover {{
        background-color: #667eea !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }}
    
    /* Settings panel buttons */
    div[data-testid="stVerticalBlock"] button {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    
    div[data-testid="stVerticalBlock"] button:hover {{
        background-color: {border_color} !important;
        transform: scale(1.02) !important;
    }}
    
    /* Help section styling */
    .help-banner {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s ease-in;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Settings panel improvements */
    .settings-section {{
        background-color: {secondary_bg};
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid {border_color};
    }}
    
    /* Status indicators with better styling */
    .status-good {{
        color: #10b981;
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 6px;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        margin-bottom: 1rem;
        border: 1px solid #10b981;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
</style>
"""

# Apply dynamic CSS based on current theme
st.markdown(get_css_styles(st.session_state.get('dark_mode', False)), unsafe_allow_html=True)

@st.cache_resource
def load_documents():
    """Load and process PDF documents"""
    docs_path = './input'
    all_docs = []
    
    if not os.path.exists(docs_path):
        st.error(f"Input directory '{docs_path}' not found!")
        return []
    
    with st.spinner("Loading documents..."):
        for file in os.listdir(docs_path):
            if file.endswith('.pdf'):
                try:
                    loader = PyPDFLoader(os.path.join(docs_path, file))
                    pages = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    docs = splitter.split_documents(pages)
                    all_docs.extend(docs)
                except Exception as e:
                    st.markdown(f'<div class="status-info">Error loading {file}: {str(e)}</div>', unsafe_allow_html=True)
    
    return all_docs

def get_documents_hash(docs_path='./input'):
    """Generate hash of all PDF files for cache validation"""
    hash_md5 = hashlib.md5()
    
    if not os.path.exists(docs_path):
        return ""
    
    pdf_files = sorted([f for f in os.listdir(docs_path) if f.endswith('.pdf')])
    
    for file in pdf_files:
        file_path = os.path.join(docs_path, file)
        if os.path.exists(file_path):
            # Include filename and modification time in hash
            file_info = f"{file}_{os.path.getmtime(file_path)}"
            hash_md5.update(file_info.encode())
    
    return hash_md5.hexdigest()

@st.cache_resource
def create_or_load_vectorstore(_docs):
    """Create or load cached FAISS vectorstore"""
    cache_dir = './faiss_cache'
    index_file = os.path.join(cache_dir, 'index.faiss')
    pkl_file = os.path.join(cache_dir, 'index.pkl')
    hash_file = os.path.join(cache_dir, 'docs_hash.txt')
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get current documents hash
    current_hash = get_documents_hash()
    
    # Check if cached vectorstore exists and is valid
    if (os.path.exists(index_file) and 
        os.path.exists(pkl_file) and 
        os.path.exists(hash_file)):
        
        try:
            # Read stored hash
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            # If hashes match, load cached vectorstore
            if stored_hash == current_hash and current_hash:
                st.info("Loading cached vector database...")
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local(cache_dir, embedding_model, allow_dangerous_deserialization=True)
                st.success("Loaded cached vectors successfully!")
                return vectorstore
        except Exception as e:
            st.markdown(f'<div class="status-info">Failed to load cached vectors: {str(e)}</div>', unsafe_allow_html=True)
    
    # Create new vectorstore if cache is invalid or doesn't exist
    if not _docs:
        return None
    
    with st.spinner("Creating new vector database..."):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(_docs, embedding_model)
        
        # Save vectorstore to cache
        try:
            vectorstore.save_local(cache_dir)
            # Save documents hash
            with open(hash_file, 'w') as f:
                f.write(current_hash)
            st.success("Vector database cached for faster future loading!")
        except Exception as e:
            st.markdown(f'<div class="status-info">Failed to cache vectors: {str(e)}</div>', unsafe_allow_html=True)
    
    return vectorstore

def check_ollama_status():
    """Check if Ollama is running and which models are available"""
    try:
        # Use local Ollama directly (much faster!)
        llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")
        # Test connection with a simple query
        llm.invoke("test")
        return True, "Local Ollama (Fast Mode)"
    except Exception as e:
        return False, str(e)

def process_deepseek_response(response_text):
    """Remove thinking process from deepseek responses and return clean answer"""
    if "deepseek" in st.session_state.get('selected_model', '').lower():
        # Remove thinking tags and content
        import re
        # Remove <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        # Remove <thinking>...</thinking> blocks
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
        # Remove any remaining thinking patterns
        cleaned = re.sub(r'\*\*Thinking:?\*\*.*?(?=\n\n|\*\*|$)', '', cleaned, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    return response_text

def create_qa_chain(model_name, vectorstore):
    """Create QA chain with specified model using local Ollama"""
    if vectorstore is None:
        return None
    
    try:
        # Connect to local Ollama for maximum speed
        llm = Ollama(
            model=model_name, 
            base_url="http://localhost:11434",
            # Optimize for speed
            num_ctx=2048,  # Smaller context for faster processing
            temperature=0.1,  # Lower temperature for faster, more focused responses
        )
        
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Reduced from 4 to 3 for speed
            llm=llm
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=multi_retriever,
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

def get_sample_questions():
    """Get top 3 sample questions for main interface"""
    return [
        "How many AI methods does the paper mention?",
        "Does Equisoft support training for agents?", 
        "What is the main contribution of the paper?"
    ]

def main():
    # Apply dynamic CSS based on current theme
    st.markdown(get_css_styles(st.session_state.get('dark_mode', False)), unsafe_allow_html=True)
    
    # Check Ollama status
    ollama_running, status_msg = check_ollama_status()
    if not ollama_running:
        st.error(f"Ollama not running: {status_msg}")
        st.info("Please start Ollama locally: `ollama serve` then refresh this page")
        st.info("Make sure you have the models: `ollama pull llama3.2:latest` and `ollama pull deepseek-r1:8b`")
        return
    
    # Load documents and create vectorstore
    docs = load_documents()
    if not docs:
        st.error("No documents loaded. Please add PDF files to the 'input' directory.")
        return
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = create_or_load_vectorstore(docs)
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'show_help' not in st.session_state:
        st.session_state.show_help = True
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    
    # Create ChatGPT-style layout with sidebar and main content
    sidebar = st.sidebar
    
    # Sidebar - Conversation History (like ChatGPT)
    with sidebar:
        st.markdown("### 💬 Conversations")
        
        # New conversation button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            # Save current conversation if it has messages
            if st.session_state.messages:
                conversation_title = st.session_state.messages[0]["content"][:30] + "..."
                st.session_state.conversation_history.append({
                    "title": conversation_title,
                    "messages": st.session_state.messages.copy(),
                    "timestamp": time.time()
                })
            # Start new conversation
            st.session_state.messages = []
            st.session_state.current_conversation = None
            st.rerun()
        
        st.markdown("---")
        
        # Display conversation history
        for i, conv in enumerate(reversed(st.session_state.conversation_history)):
            if st.button(
                f"📄 {conv['title']}", 
                key=f"conv_{i}",
                use_container_width=True,
                help=f"Created {time.strftime('%Y-%m-%d %H:%M', time.localtime(conv['timestamp']))}"
            ):
                # Save current conversation before switching
                if st.session_state.messages and st.session_state.current_conversation is None:
                    conversation_title = st.session_state.messages[0]["content"][:30] + "..."
                    st.session_state.conversation_history.append({
                        "title": conversation_title,
                        "messages": st.session_state.messages.copy(),
                        "timestamp": time.time()
                    })
                # Load selected conversation
                st.session_state.messages = conv['messages'].copy()
                st.session_state.current_conversation = i
                st.rerun()
        
        # Clear all history button
        if st.session_state.conversation_history:
            st.markdown("---")
            if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                st.session_state.conversation_history = []
                st.rerun()
    
    # Main content area
    # Top bar with settings (like ChatGPT)
    col1, col2, col3 = st.columns([0.65, 0.25, 0.1])
    
    with col1:
        st.markdown('<h1 class="main-header">🧠 DocuMind AI</h1>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-good">Connected to Local Server • {len(docs)} Documents Loaded</div>', unsafe_allow_html=True)
    
    with col2:
        # Model selection in top bar
        model_options = ["llama3.2:latest", "deepseek-r1:8b", "llama3.1:8b", "gemma2:9b"]
        selected_model = st.selectbox(
            "AI Model",
            model_options,
            index=0,
            help="🤖 AI Model: llama3.2 (Balanced) • deepseek (Reasoning) • llama3.1 (Reliable) • gemma2 (Fast)",
            key="model_selection_main",
            label_visibility="collapsed"
        )
        st.session_state.selected_model = selected_model
    
    with col3:
        # Settings dropdown
        with st.popover("⚙️", help="Settings & Preferences", use_container_width=True):
            st.markdown("**⚙️ DocuMind Settings**")
            
            # Theme toggle
            theme_text = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
            if st.button(theme_text, use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
            
            # Sources toggle
            sources_text = "📄 Hide Sources" if st.session_state.show_sources else "📄 Show Sources"
            if st.button(f"� {sources_text}", use_container_width=True):
                st.session_state.show_sources = not st.session_state.show_sources
                st.rerun()
            
            # Help toggle
            help_text = "❓ Hide Help" if st.session_state.show_help else "❓ Show Help"
            if st.button(help_text, use_container_width=True):
                st.session_state.show_help = not st.session_state.show_help
                st.rerun()
            
            st.markdown("---")
            st.markdown(f"**📊 Analytics**")
            st.markdown(f"• Documents: {len(docs)}")
            st.markdown(f"• Conversations: {len(st.session_state.conversation_history)}")
            
            # About
            with st.expander("ℹ️ About DocuMind AI"):
                st.markdown("**🧠 DocuMind AI v2.0**")
                st.markdown("Intelligent Document Assistant")
                st.markdown("**Developer:** Jason Xu")
                st.markdown("📧 nvtech.consult@gmail.com")
    
    # Main chat area
    # Help section (shown when help is enabled)
    if st.session_state.show_help and len(st.session_state.messages) == 0:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
            <h3>🧠 Welcome to DocuMind AI</h3>
            <p><strong>Intelligent Document Assistant - How to use:</strong></p>
            <ul>
                <li>📄 <strong>Ask questions</strong> about your uploaded documents</li>
                <li>🤖 <strong>Choose your AI model</strong> in the top bar</li>
                <li>🌙 <strong>Toggle themes</strong> in settings (⚙️)</li>
                <li>📋 <strong>Show/hide sources</strong> in settings</li>
            </ul>
            <p><em>💡 Tip: Start with the sample questions below, or ask anything about your documents!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample questions section (above chat)
    if len(st.session_state.messages) == 0:
        if st.session_state.show_help:
            st.markdown("### 💡 Try these sample questions:")
        else:
            st.markdown("### Ask me anything about your documents:")
        
        sample_questions = get_sample_questions()
        cols = st.columns(3)
        
        for i, question in enumerate(sample_questions):
            with cols[i]:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    st.session_state.current_question = question
    
    # Chat messages container with proper spacing for fixed input
    st.markdown('<div class="chat-container" style="padding-bottom: 120px;">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Agent</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if available and enabled
            if "sources" in message and st.session_state.show_sources:
                sources_html = "<div class='source-docs'><strong>📄 Sources:</strong><br>"
                for source in message["sources"]:
                    sources_html += f"• {source}<br>"
                sources_html += "</div>"
                st.markdown(sources_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Fixed chat input at the bottom (outside of columns, full width)
    # Enhanced chat input with guidance
    if st.session_state.show_help and len(st.session_state.messages) == 0:
        chat_placeholder = "💬 Type your question here... (e.g., 'What are the main findings?' or 'Summarize the key points')"
    else:
        chat_placeholder = "Ask a question about your documents..."
        
    question = st.chat_input(chat_placeholder)
    
    # Handle sample question selection
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
    
    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get answer with timing
        start_time = time.time()
        with st.spinner(f"Processing with {selected_model}..."):
            qa_chain = create_qa_chain(selected_model, st.session_state.vectorstore)
            
            if qa_chain:
                try:
                    result = qa_chain(question)
                    end_time = time.time()
                    response_time = f"{end_time - start_time:.1f}s"
                    
                    # Process the answer and clean deepseek responses
                    raw_answer = result["result"]
                    answer = process_deepseek_response(raw_answer)
                    sources = []
                    
                    # Enhanced source formatting
                    for doc in result.get("source_documents", []):
                        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        page = doc.metadata.get('page', 'Unknown')
                        sources.append(f"📄 {source_name} (Page: {page})")
                    
                    # Enhanced answer formatting
                    formatted_answer = f"{answer}"
                    if st.session_state.show_help:
                        formatted_answer += f"\n\n⚡ *Response time: {response_time} using {selected_model}*"
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    if st.session_state.show_help:
                        error_msg += "\n\n💡 **Troubleshooting:**\n- Make sure Ollama is running: `ollama serve`\n- Check if the model is installed: `ollama list`\n- Try a different model from the settings"
                    st.error(error_msg)
            else:
                st.error("Failed to create QA chain. Please check Ollama connection.")
        
        # Rerun to show new messages
        st.rerun()

if __name__ == "__main__":
    main()
