#!/bin/bash

# DocuMind AI - Startup Script
echo "🧠 Starting DocuMind AI..."
echo "======================================"

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Warning: Ollama service not detected"
    echo "📝 Please start Ollama first: 'ollama serve'"
    echo ""
fi

# Check if required Python packages are installed
echo "📦 Checking dependencies..."
python -c "import streamlit, langchain" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

echo "✅ Dependencies verified"
echo ""

# Start the DocuMind AI application
echo "🚀 Launching DocuMind AI on http://localhost:8501"
echo "💡 Press Ctrl+C to stop the application"
echo ""

streamlit run app_fast.py --server.port 8501 --server.address 0.0.0.0
