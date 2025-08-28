# app_fast.py
import os
import time
import random
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(
    page_title="Do        # Rotating placeholder - adapt based on current mode
        has_vectorstore = 'vectorstore' in st.session_state and st.session_state.vectorstore is not None
        force_general = st.session_state.get('force_general_mode', False)
        using_documents = has_vectorstore and not force_general
        
        if using_documents:
            hints = [
                "Ask about key findings or methods…",
                "Try: \"Summarize the main conclusions\"",
                "Ask: \"What evidence supports this claim?\"",
                "Try: \"Compare different approaches mentioned\"",
                "Ask: \"What are the limitations discussed?\"",
            ]
        else:
            hints = [
                "Ask any question…",
                "Try: \"python bubble sort algorithm\"",
                "Ask: \"explain machine learning\"",
                "Try: \"write a React component\"",
                "Ask: \"how does photosynthesis work?\"",
            ] AI - Intelligent Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------
# Styles (Light theme only)
# --------------------------
def get_css_styles():
    bg_color = "#ffffff"
    text_color = "#111827"
    secondary_bg = "#f8fafc"
    border_color = "#e5e7eb"
    user_msg_bg = "#ffffff"
    assistant_msg_bg = "#f9fafb"

    return f"""
<style>
  /* Hide Streamlit chrome */
  #MainMenu {{visibility:hidden;}}
  footer {{visibility:hidden;}}
  header {{visibility:hidden;}}

  .stApp {{
    background:{bg_color};
    color:{text_color};
    padding-bottom:100px !important;
  }}

  .block-container {{
    max-width: 1200px;
    padding-top: 0.75rem;
  }}

  /* Header styling */
  .main-header {{
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
    color: {text_color};
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  .status-good {{
    color: #065f46;
    font-weight: 500;
    font-size: 0.92rem;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    margin: 0.5rem 0 1rem 0;
    animation: pulse-green 2s infinite;
  }}

  @keyframes pulse-green {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.8; }}
  }}

  /* Welcome banner */
  .welcome {{
    background: linear-gradient(135deg, #e0f2fe 0%, #ede9fe 100%);
    color: #1f2937;
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid {border_color};
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }}

  /* Enhanced button styling */
  .stButton > button {{
    background: {secondary_bg} !important;
    color: {text_color} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
  }}

  .stButton > button:hover {{
    background: #667eea !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102,126,234,0.3) !important;
  }}

  /* Chat messages */
  .chat-message {{
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    border-radius: 12px;
    border: 1px solid {border_color};
    font-size: 1rem;
    line-height: 1.6;
    color: {text_color};
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .user-message {{ 
    background: {user_msg_bg}; 
    border-left: 4px solid #667eea;
  }}
  .assistant-message {{ 
    background: {assistant_msg_bg}; 
    border-left: 4px solid #10b981;
  }}

  .source-inline {{
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: {secondary_bg};
    border-radius: 6px;
    border: 1px solid {border_color};
  }}

  /* Chat input improvements */
  .stChatInput {{
    position: fixed !important;
    bottom: 0 !important; left: 0 !important; right: 0 !important;
    background: {bg_color} !important;
    border-top: 1px solid {border_color} !important;
    padding: 1rem !important;
    z-index: 999 !important;
    box-shadow: 0 -4px 12px rgba(0,0,0,0.1) !important;
  }}
  .stChatInput > div {{
    max-width: 900px !important;
    margin: 0 auto !important;
  }}

  /* Settings popover styling */
  .stPopover {{
    border-radius: 8px !important;
  }}

  /* Spinner customization */
  .stSpinner > div {{
    border-color: #667eea transparent #667eea transparent !important;
  }}

  /* Expander improvements */
  .streamlit-expanderHeader {{
    background-color: {secondary_bg} !important;
    border-radius: 8px !important;
    border: 1px solid {border_color} !important;
  }}

  /* Loading states */
  .loading-indicator {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    background: {secondary_bg};
    border-radius: 8px;
    border: 1px solid {border_color};
    font-size: 0.9rem;
    color: #6b7280;
  }}
</style>
"""

st.markdown(get_css_styles(), unsafe_allow_html=True)


# --------------------------
# Data loading & Vector store
# --------------------------
@st.cache_resource
def load_documents():
    docs_path = "./input"
    all_docs = []
    if not os.path.exists(docs_path):
        return []
    for file in os.listdir(docs_path):
        if file.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(docs_path, file))
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(pages)
                all_docs.extend(docs)
            except Exception as e:
                st.warning(f"Error loading {file}: {e}")
    return all_docs


def get_documents_hash(docs_path="./input"):
    hash_md5 = hashlib.md5()
    if not os.path.exists(docs_path):
        return ""
    pdf_files = sorted([f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")])
    for f in pdf_files:
        fp = os.path.join(docs_path, f)
        if os.path.exists(fp):
            info = f"{f}_{os.path.getmtime(fp)}"
            hash_md5.update(info.encode())
    return hash_md5.hexdigest()


@st.cache_resource
def create_or_load_vectorstore(_docs):
    cache_dir = "./faiss_cache"
    index_file = os.path.join(cache_dir, "index.faiss")
    pkl_file = os.path.join(cache_dir, "index.pkl")
    hash_file = os.path.join(cache_dir, "docs_hash.txt")
    os.makedirs(cache_dir, exist_ok=True)

    current_hash = get_documents_hash()

    if os.path.exists(index_file) and os.path.exists(pkl_file) and os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                stored_hash = f.read().strip()
            if stored_hash == current_hash and current_hash:
                emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vs = FAISS.load_local(cache_dir, emb, allow_dangerous_deserialization=True)
                return vs
        except Exception:
            pass

    if not _docs:
        return None
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(_docs, emb)
    vs.save_local(cache_dir)
    with open(hash_file, "w") as f:
        f.write(current_hash)
    return vs


# --------------------------
# LLM / QA chain utilities
# --------------------------
def check_ollama_status():
    try:
        llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")
        llm.invoke("ping")
        return True, "Local Ollama OK"
    except Exception as e:
        return False, str(e)


def process_deepseek_response(text):
    if "deepseek" in st.session_state.get("selected_model", "").lower():
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"\*\*Thinking:?\*\*.*?(?=\n\n|\*\*|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    return text.strip()


def build_llm(model_name, fast_mode=False):
    """Create LLM with optimized settings for speed and quality."""
    threads = max(1, (os.cpu_count() or 4))
    
    # Check if detailed responses are enabled
    detailed_mode = st.session_state.get('detailed_mode', True)
    
    # Token limits based on mode settings
    if fast_mode and not detailed_mode:
        # Fast mode with short responses
        num_ctx = 2048
        num_predict = 512
    elif fast_mode and detailed_mode:
        # Fast mode but allow detailed responses
        num_ctx = 3072
        num_predict = 1536
    elif not fast_mode and detailed_mode:
        # Standard mode with detailed responses (recommended)
        num_ctx = 4096
        num_predict = 2048
    else:
        # Standard mode with normal responses
        num_ctx = 2048
        num_predict = 1024
    
    # Better parameters for response quality
    temperature = 0.3       # Slightly higher for more natural responses
    top_p = 0.9
    repeat_penalty = 1.1

    return Ollama(
        model=model_name,
        base_url="http://localhost:11434",
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        num_thread=threads,
    )


def create_general_qa_chain(model_name, fast_mode=False):
    """Create a general-purpose QA chain for non-document questions."""
    try:
        llm = build_llm(model_name, fast_mode=fast_mode)
        
        # General knowledge prompt template
        prompt_template = """You are a helpful AI assistant. Answer the user's question with detailed explanations and examples when appropriate.

Question: {question}

Instructions:
- Provide a comprehensive and detailed answer
- Include examples, code snippets, or step-by-step explanations when relevant
- Be accurate and informative
- If it's a programming question, provide working code examples
- If you're not certain about something, mention it

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["question"]
        )
        
        # Create a simple LLM chain for general questions
        chain = LLMChain(llm=llm, prompt=PROMPT)
        return chain
        
    except Exception as e:
        st.error(f"Error creating general QA chain: {e}")
        return None


def create_qa_chain(model_name, vectorstore, fast_mode=False):
    if vectorstore is None:
        return None
    try:
        llm = build_llm(model_name, fast_mode=fast_mode)

        # Enhanced prompt template for better, more complete responses
        prompt_template = """You are an intelligent document assistant. Use the following context to provide comprehensive and detailed answers to the user's question.

Context: {context}

Question: {question}

Instructions:
- Provide a complete and thorough answer based on the context
- Include specific details, examples, and explanations when available
- If the context contains multiple relevant points, address them all
- Be specific and cite information from the documents when possible
- If information is incomplete, explain what is available and what might be missing
- Do not truncate your response - provide a full, comprehensive answer

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        # Choose retrieval strategy based on mode
        if fast_mode or model_name.lower().startswith("deepseek"):
            # Fast mode: direct retrieval with more documents for better context
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Increased from 2
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=retriever, 
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        else:
            # Standard mode: multi-query for better retrieval
            base_ret = vectorstore.as_retriever(search_kwargs={"k": 5})  # Increased from 3
            mqr = MultiQueryRetriever.from_llm(retriever=base_ret, llm=llm)
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=mqr, 
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        return qa
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None


def get_sample_questions():
    return [
        "How many AI methods does the paper \"traffic light detection\" mention?",
        "Does Equisoft support training for agents?",
        "How many insurance companies does equisoft partner with?",
    ]


# --------------------------
# App
# --------------------------
def main():
    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "show_help" not in st.session_state:
        st.session_state.show_help = True
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama3.2:latest"
    if "fast_mode" not in st.session_state:
        st.session_state.fast_mode = True  # turn on by default for speed

    # Check LLM backend
    ok, status = check_ollama_status()
    if not ok:
        st.error(f"Ollama not running: {status}")
        st.info("Start with:  `ollama serve`")
        st.info("Install models:  `ollama pull llama3.2:latest` and/or `ollama pull deepseek-r1:8b`")
        return

    docs = load_documents()
    
    # Initialize vectorstore if documents are available
    if docs:
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = create_or_load_vectorstore(docs)
        if st.session_state.vectorstore is None:
            st.warning("Failed to prepare vector database. Switching to general Q&A mode.")
    else:
        # No documents - show info but continue in general mode
        st.info("💡 No documents found in ./input. You can still ask general questions! Add PDFs for document-specific Q&A.")
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore  # Clear any existing vectorstore

    # Layout columns: left history (slim) + main
    col1, col2 = st.columns([0.18, 0.82])

    # Left: Chat history (collapsible)
    with col1:
        with st.expander("💬 Chat History", expanded=True):
            if st.button("➕ New Chat", use_container_width=True):
                if st.session_state.messages:
                    title = st.session_state.messages[0]["content"][:30] + "..."
                    st.session_state.conversation_history.append({
                        "title": title,
                        "messages": st.session_state.messages.copy(),
                        "timestamp": time.time(),
                    })
                st.session_state.messages = []
                st.session_state.current_conversation = None
                st.rerun()

            st.markdown("---")

            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                if st.button(f"📄 {conv['title']}", key=f"conv_{i}", use_container_width=True):
                    if st.session_state.messages and st.session_state.current_conversation is None:
                        title = st.session_state.messages[0]["content"][:30] + "..."
                        st.session_state.conversation_history.append({
                            "title": title,
                            "messages": st.session_state.messages.copy(),
                            "timestamp": time.time(),
                        })
                    st.session_state.messages = conv["messages"].copy()
                    st.session_state.current_conversation = i
                    st.rerun()

            if st.session_state.conversation_history:
                st.markdown("---")
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.conversation_history = []
                    st.rerun()
            
            # Document list section
            if docs:
                st.markdown("---")
                with st.expander("📚 Loaded Documents", expanded=False):
                    pdf_files = []
                    docs_path = "./input"
                    if os.path.exists(docs_path):
                        pdf_files = sorted([f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")])
                    
                    if pdf_files:
                        st.markdown(f"**{len(pdf_files)} PDF files:**")
                        for pdf_file in pdf_files:
                            # Remove .pdf extension for display
                            display_name = pdf_file.replace('.pdf', '')
                            st.markdown(f"📄 {display_name}")
                    else:
                        st.markdown("No documents found")
            else:
                st.markdown("---")
                st.info("💡 Add PDF files to `./input` folder to enable document Q&A")
                

    # Right: Main content
    with col2:
        # Top bar: title + settings icon (popover)
        top_l, top_r = st.columns([0.94, 0.06])
        with top_l:
            has_vectorstore = 'vectorstore' in st.session_state and st.session_state.vectorstore is not None
            force_general = st.session_state.get('force_general_mode', False)
            doc_count = len(docs) if docs else 0
            
            if has_vectorstore and not force_general:
                mode_text = f"📚 Document Mode • {doc_count} Files"
            elif has_vectorstore and force_general:
                mode_text = f"🧠 General Mode • {doc_count} Files Available"
            else:
                mode_text = "🧠 General Q&A Mode"
            
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between;">
                  <h1 class="main-header">🧠 DocuMind AI</h1>
                </div>
                <div class="status-good">✅ Connected to Local Server • {mode_text}</div>
                """,
                unsafe_allow_html=True,
            )
        with top_r:
            with st.popover("⚙️", help="Settings"):
                st.markdown("**Settings**")
                model_options = [
                    "llama3.2:latest",    # 2.0 GB - Fast and lightweight
                    "gemma2:9b",          # 5.4 GB - Balanced performance
                    "deepseek-r1:8b",     # 5.2 GB - Best for reasoning
                ]
                # keep current if present
                cur = st.session_state.get("selected_model", "llama3.2:latest")
                idx = model_options.index(cur) if cur in model_options else 0
                st.session_state.selected_model = st.selectbox(
                    "AI Model",
                    model_options,
                    index=idx,
                    help="💡 Llama3.2 (2GB): Fastest responses. Gemma2 (5.4GB): Balanced speed/quality. DeepSeek-R1 (5.2GB): Best reasoning and complex analysis.",
                    key="model_selection_settings",
                )

                st.session_state.fast_mode = st.toggle(
                    "⚡ Fast mode (shorter, faster)", 
                    value=st.session_state.fast_mode,
                    help="Enable for quicker responses with shorter context"
                )
                
                # New toggle for detailed responses
                if "detailed_mode" not in st.session_state:
                    st.session_state.detailed_mode = True
                    
                st.session_state.detailed_mode = st.toggle(
                    "📝 Detailed responses (recommended)", 
                    value=st.session_state.detailed_mode,
                    help="Allow longer, more comprehensive answers"
                )
                
                # Mode selection toggle (only show if documents are available)
                has_docs = 'vectorstore' in st.session_state and st.session_state.vectorstore is not None
                if has_docs:
                    if "force_general_mode" not in st.session_state:
                        st.session_state.force_general_mode = False
                    
                    st.session_state.force_general_mode = st.toggle(
                        "🧠 General Q&A Mode", 
                        value=st.session_state.force_general_mode,
                        help="Enable to ignore documents and use general AI knowledge instead"
                    )
                    
                    if st.session_state.force_general_mode:
                        st.info("💡 Mode: General Q&A (ignoring documents)")
                    else:
                        st.info("📚 Mode: Document Q&A (using uploaded documents)")

                src_btn = "📄 Hide Sources" if st.session_state.show_sources else "📄 Show Sources"
                if st.button(src_btn, use_container_width=True):
                    st.session_state.show_sources = not st.session_state.show_sources
                    st.rerun()

                help_btn = "❓ Hide Help" if st.session_state.show_help else "❓ Show Help"
                if st.button(help_btn, use_container_width=True):
                    st.session_state.show_help = not st.session_state.show_help
                    st.rerun()

        # Welcome banner
        if st.session_state.show_help and len(st.session_state.messages) == 0:
            has_vectorstore = 'vectorstore' in st.session_state and st.session_state.vectorstore is not None
            force_general = st.session_state.get('force_general_mode', False)
            
            if has_vectorstore and not force_general:
                # Document mode
                welcome_content = """
                <div class="welcome">
                  <strong>🤖 Welcome to DocuMind AI</strong>
                  <p style="margin: 0.5rem 0;">Your intelligent document assistant powered by local AI models.</p>
                  <ul style="margin:0.5rem 0 0 1.25rem; padding-left: 0;">
                    <li>📚 Ask questions about your uploaded PDFs</li>
                    <li>📝 Detailed responses are enabled by default for comprehensive answers</li>
                    <li>⚡ Use Fast mode for quicker responses when needed</li>
                    <li>🧠 DeepSeek excels at complex reasoning (may take longer)</li>
                    <li>📄 Toggle sources to see document references</li>
                    <li>🔄 Switch to General Q&A mode in settings to ignore documents</li>
                  </ul>
                  <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; color: #6b7280;">
                    💡 <strong>Tip:</strong> Try asking specific questions about your documents for better results!
                  </p>
                </div>
                """
            else:
                # General mode (either no docs or forced general)
                mode_reason = " (documents ignored)" if has_vectorstore and force_general else ""
                welcome_content = f"""
                <div class="welcome">
                  <strong>🤖 Welcome to DocuMind AI</strong>
                  <p style="margin: 0.5rem 0;">Your intelligent AI assistant powered by local models{mode_reason}.</p>
                  <ul style="margin:0.5rem 0 0 1.25rem; padding-left: 0;">
                    <li>🧠 Ask any general questions (programming, math, explanations, etc.)</li>
                    <li>💻 Get code examples and programming help</li>
                    <li>📝 Detailed responses are enabled by default</li>
                    <li>⚡ Use Fast mode for quicker responses</li>
                    {"<li>📚 Switch to Document Q&A mode in settings to use uploaded documents</li>" if has_vectorstore else "<li>📚 Add PDFs to ./input folder for document-specific Q&A</li>"}
                  </ul>
                  <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; color: #6b7280;">
                    💡 <strong>Examples:</strong> "python bubble sort", "explain quantum physics", "write a React component"
                  </p>
                </div>
                """
            
            st.markdown(welcome_content, unsafe_allow_html=True)

        # Sample question pills
        if len(st.session_state.messages) == 0:
            st.markdown("### 💡 Try these sample questions:")
            qcols = st.columns(3)
            samples = get_sample_questions()
            for i, q in enumerate(samples):
                with qcols[i]:
                    if st.button(f"💡 {q}", use_container_width=True, key=f"sample_{i}"):
                        st.session_state.current_question = q
                        st.rerun()

        # Messages
        for msg in st.session_state.messages:
            role = msg["role"]
            cls = "user-message" if role == "user" else "assistant-message"
            st.markdown(
                f'<div class="chat-message {cls}"><strong>{"You" if role=="user" else "AI Agent"}</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if role != "user" and st.session_state.show_sources and msg.get("sources"):
                if msg["sources"]:  # Only show if there are actually sources
                    sources_html = '<div class="source-inline">'
                    sources_html += '<strong>� Sources:</strong><br>'
                    for source in msg["sources"]:
                        sources_html += f'&nbsp;&nbsp;• {source}<br>'
                    sources_html += '</div>'
                    st.markdown(sources_html, unsafe_allow_html=True)

        # Rotating placeholder
        hints = [
            "Ask about findings or methods…",
            "Try: “Summarize the key points.”",
            "Ask: “Which pages support this?”",
        ]

        # Auto-submit if a sample was clicked
        if "current_question" in st.session_state:
            question = st.session_state.current_question
            del st.session_state.current_question
        else:
            question = st.chat_input(random.choice(hints))

        # Handle question
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            start = time.time()
            
            # Determine if we have documents or need general mode
            has_vectorstore = 'vectorstore' in st.session_state and st.session_state.vectorstore is not None
            force_general = st.session_state.get('force_general_mode', False)
            has_documents = has_vectorstore and not force_general
            
            # Enhanced loading message with model info
            mode_info = "Document QA" if has_documents else "General QA"
            loading_msg = f"🤔 Processing with {st.session_state.selected_model} ({mode_info})"
            if st.session_state.fast_mode:
                loading_msg += " (Fast Mode)"
            loading_msg += "…"
            
            with st.spinner(loading_msg):
                if has_documents:
                    # Document-based QA
                    qa = create_qa_chain(
                        st.session_state.selected_model,
                        st.session_state.vectorstore,
                        fast_mode=st.session_state.fast_mode
                    )
                    if qa is None:
                        st.error("⚠️ QA chain unavailable. Please check your model configuration.")
                    else:
                        try:
                            result = qa(question)
                            elapsed = f"{time.time() - start:.1f}s"
                            raw = result.get("result", "")
                            
                            # Better error checking for empty responses
                            if not raw or raw.strip() == "":
                                raw = "I couldn't find a relevant answer in the uploaded documents. Please try rephrasing your question or check if the information is available in the documents."
                            
                            answer = process_deepseek_response(raw)
                            
                            # Check if response seems truncated and detailed mode is enabled
                            if (st.session_state.get('detailed_mode', True) and 
                                len(answer.split()) < 50 and 
                                not answer.endswith(('.', '!', '?')) and 
                                'couldn\'t find' not in answer.lower()):
                                
                                # Try to get a more complete response with a follow-up
                                try:
                                    follow_up = f"{question}\n\nPlease provide a complete and detailed answer with full explanations."
                                    result2 = qa(follow_up)
                                    raw2 = result2.get("result", "")
                                    if raw2 and len(raw2.strip()) > len(answer.strip()):
                                        answer = process_deepseek_response(raw2)
                                        # Update sources if better result
                                        if result2.get("source_documents"):
                                            result = result2
                                except:
                                    pass  # If follow-up fails, use original answer

                            sources = []
                            for d in result.get("source_documents", []):
                                source_path = d.metadata.get("source", "Unknown")
                                name = os.path.basename(source_path)
                                # Remove .pdf extension for cleaner display
                                clean_name = name.replace('.pdf', '') if name.endswith('.pdf') else name
                                page = d.metadata.get("page", "—")
                                sources.append(f"📄 {clean_name} (page {page})")

                            content = answer
                            if st.session_state.show_help:
                                speed_indicator = "📚" if has_documents else "🧠"
                                content += f"\n\n_{speed_indicator} Response time: {elapsed} (mode: {'doc' if has_documents else 'general'}, fast={st.session_state.fast_mode})_"

                            st.session_state.messages.append(
                                {"role": "assistant", "content": content, "sources": sources}
                            )
                        except Exception as e:
                            error_details = str(e)
                            if "connection" in error_details.lower():
                                st.error(f"� Connection Error: Unable to connect to the AI model. Please check if Ollama is running.")
                            elif "model" in error_details.lower():
                                st.error(f"🤖 Model Error: Issue with {st.session_state.selected_model}. Try switching to a different model.")
                            else:
                                st.error(f"❌ Error: {error_details}")
                            
                            # Add error message to chat history for context
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"Sorry, I encountered an error: {error_details}", "sources": []}
                            )
                else:
                    # General-purpose QA (no documents)
                    general_qa = create_general_qa_chain(
                        st.session_state.selected_model,
                        fast_mode=st.session_state.fast_mode
                    )
                    if general_qa is None:
                        st.error("⚠️ General QA chain unavailable. Please check your model configuration.")
                    else:
                        try:
                            result = general_qa.run(question)
                            elapsed = f"{time.time() - start:.1f}s"
                            
                            if not result or result.strip() == "":
                                result = "I'm sorry, I couldn't generate a response to your question. Please try rephrasing it."
                            
                            answer = process_deepseek_response(result)
                            
                            content = answer
                            if st.session_state.show_help:
                                content += f"\n\n_🧠 Response time: {elapsed} (general mode, fast={st.session_state.fast_mode})_"

                            st.session_state.messages.append(
                                {"role": "assistant", "content": content, "sources": []}
                            )
                        except Exception as e:
                            error_details = str(e)
                            if "connection" in error_details.lower():
                                st.error(f"🔌 Connection Error: Unable to connect to the AI model. Please check if Ollama is running.")
                            elif "model" in error_details.lower():
                                st.error(f"🤖 Model Error: Issue with {st.session_state.selected_model}. Try switching to a different model.")
                            else:
                                st.error(f"❌ Error: {error_details}")
                            
                            # Add error message to chat history for context
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"Sorry, I encountered an error: {error_details}", "sources": []}
                            )

            st.rerun()


if __name__ == "__main__":
    main()