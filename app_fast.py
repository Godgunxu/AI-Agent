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


# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(
    page_title="DocuMind AI - Intelligent Document Assistant",
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
    secondary_bg = "#f6f7f8"
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
    padding-bottom:100px !important; /* room for fixed chat input */
  }}

  .block-container {{
    max-width: 1200px;
    padding-top: 0.75rem;
  }}

  /* Top header */
  .main-header {{
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0;
    color: {text_color};
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
  }}

  /* Welcome banner (light, compact) */
  .welcome {{
    background: linear-gradient(135deg, #e0f2fe 0%, #ede9fe 100%);
    color: #1f2937;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid {border_color};
    margin-bottom: 1rem;
  }}

  /* Sample question pills */
  .pill {{
    display:inline-block;
    background:#fff;
    color:{text_color};
    border:1px solid {border_color};
    border-radius:9999px;
    padding:0.55rem 0.9rem;
    font-size:0.95rem;
    margin:0.25rem 0;
    transition: background 0.15s ease, transform 0.15s ease;
  }}
  .pill:hover {{
    background:#f3f4f6;
    transform: translateY(-1px);
  }}

  /* Chat bubbles */
  .chat-message {{
    padding: 0.85rem 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    border: 1px solid {border_color};
    font-size: 1rem;
    line-height: 1.5;
    color: {text_color};
  }}
  .user-message {{ background: {user_msg_bg}; }}
  .assistant-message {{ background: {assistant_msg_bg}; }}

  .source-inline {{
    font-size: 0.9rem;
    color: #374151;
    margin-top: 0.35rem;
  }}

  /* Fixed chat input style */
  .stChatInput {{
    position: fixed !important;
    bottom: 0 !important; left: 0 !important; right: 0 !important;
    background: {bg_color} !important;
    border-top: 1px solid {border_color} !important;
    padding: 0.75rem 1rem !important;
    z-index: 999 !important;
  }}
  .stChatInput > div {{
    max-width: 900px !important;
    margin: 0 auto !important;
  }}
</style>
"""


# Force light UI
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

    # Try load
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

    # Build fresh
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


def create_qa_chain(model_name, vectorstore):
    if vectorstore is None:
        return None
    try:
        llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",
            num_ctx=2048,
            temperature=0.1,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=mqr, return_source_documents=True)
        return qa
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None


def get_sample_questions():
    return [
        "How many AI methods does the paper mention?",
        "Does Equisoft support training for agents?",
        "What is the main contribution of the paper?",
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

    # Check LLM backend
    ok, status = check_ollama_status()
    if not ok:
        st.error(f"Ollama not running: {status}")
        st.info("Start with:  `ollama serve`")
        st.info("Install models:  `ollama pull llama3.2:latest` and/or `ollama pull deepseek-r1:8b`")
        return

    docs = load_documents()
    if not docs:
        st.error("No documents found in ./input. Please add PDFs and refresh.")
        return

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = create_or_load_vectorstore(docs)
    if st.session_state.vectorstore is None:
        st.error("Failed to prepare vector database.")
        return

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
                    # Save current before switch (if never saved)
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

    # Right: Main content
    with col2:
        # Top bar: title + settings icon (popover)
        top_l, top_r = st.columns([0.94, 0.06])
        with top_l:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between;">
                  <h1 class="main-header">🧠 DocuMind AI</h1>
                </div>
                <div class="status-good">✅ Connected to Local Server • {len(docs)} Documents Loaded</div>
                """,
                unsafe_allow_html=True,
            )
        with top_r:
            with st.popover("⚙️", help="Settings"):
                st.markdown("**Settings**")
                model_options = ["llama3.2:latest", "deepseek-r1:8b", "llama3.1:8b", "gemma2:9b"]
                st.session_state.selected_model = st.selectbox(
                    "AI Model",
                    model_options,
                    index=model_options.index(st.session_state.get("selected_model", "llama3.2:latest"))
                    if st.session_state.get("selected_model", "llama3.2:latest") in model_options else 0,
                    help="llama3.2 (Balanced) • deepseek (Reasoning) • llama3.1 (Reliable) • gemma2 (Fast)",
                    key="model_selection_settings",
                )

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
            st.markdown(
                """
                <div class="welcome">
                  <strong>👋 Welcome to DocuMind AI</strong>
                  <ul style="margin:0.35rem 0 0 1.25rem;">
                    <li>Ask questions about your uploaded PDFs</li>
                    <li>Pick a model in ⚙️ if needed</li>
                    <li>Try a sample question below</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
                inline = " • ".join(msg["sources"])
                st.markdown(f'<div class="source-inline">📄 {inline}</div>', unsafe_allow_html=True)

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
            with st.spinner(f"Processing with {st.session_state.selected_model}…"):
                qa = create_qa_chain(st.session_state.selected_model, st.session_state.vectorstore)
                if qa is None:
                    st.error("QA chain unavailable.")
                else:
                    try:
                        result = qa(question)
                        elapsed = f"{time.time() - start:.1f}s"
                        raw = result.get("result", "")
                        answer = process_deepseek_response(raw)

                        sources = []
                        for d in result.get("source_documents", []):
                            name = os.path.basename(d.metadata.get("source", "Unknown"))
                            page = d.metadata.get("page", "—")
                            sources.append(f"{name} (p.{page})")

                        content = answer
                        if st.session_state.show_help:
                            content += f"\n\n_⚡ Response time: {elapsed} using {st.session_state.selected_model}_"

                        st.session_state.messages.append(
                            {"role": "assistant", "content": content, "sources": sources}
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            st.rerun()


if __name__ == "__main__":
    main()