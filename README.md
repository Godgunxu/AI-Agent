# 🧠 DocuMind AI - Intelligent Document Assistant

A powerful AI-powered document analysis system that combines local LLM capabilities with advanced retrieval techniques for comprehensive document Q&A and general knowledge assistance.

## ✨ Features

- **Document Q&A Mode**: Analyze uploaded PDFs with context-aware responses
- **General Q&A Mode**: Answer general knowledge questions using local LLMs
- **Multi-Model Support**: Works with Llama 3.2, Gemma2 9B, and DeepSeek-R1 8B
- **Source Attribution**: Clear citations with filename and page references
- **Clean Interface**: Professional light theme with real-time processing

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM Engine**: Ollama (local deployment)
- **Vector Database**: FAISS with HuggingFace embeddings
- **Document Processing**: LangChain + PyPDF

## 📦 Installation

### Prerequisites
- Python 3.8+
- Ollama installed and running

### Setup
```bash
# Clone repository
git clone https://github.com/Godgunxu/AI-Agent.git
cd "AI Agent"

# Install dependencies
pip install -r requirements.txt

# Install Ollama models
ollama pull llama3.2:latest
ollama pull gemma2:9b
ollama pull deepseek-r1:8b

# Start Ollama service
ollama serve
```

## 🚀 Quick Start

```bash
# Use startup script (recommended)
chmod +x start_app.sh
./start_app.sh

# Or start directly
streamlit run app_fast.py --server.port 8502
```

Access the app at `http://localhost:8502`

## 💡 Usage

1. **Upload PDFs** in the sidebar
2. **Select AI model** (DeepSeek-R1 recommended for best reasoning)
3. **Choose mode**: Document Q&A or General Q&A
4. **Ask questions** and get detailed responses with sources

### Example Queries
- "What were the key points discussed in the Equisoft meeting?"
- "Explain the traffic light detection methodology"
- "What is machine learning?" (General mode)

## 📁 Project Structure

```
AI Agent/
├── Demo.ipynb              # Jupyter demonstration notebook
├── app_fast.py             # Main Streamlit application
├── start_app.sh           # Quick start script
├── requirements.txt        # Dependencies
├── .streamlit/config.toml  # Theme configuration
├── faiss_index/           # Vector database
├── faiss_cache/           # Performance cache
└── input/                 # PDF documents
```

##  Troubleshooting

**Ollama Connection Error**:
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart if needed
ollama serve
```

**Memory Issues**: Use smaller models or enable fast mode in settings.

## 📝 License

MIT License - see LICENSE file for details.