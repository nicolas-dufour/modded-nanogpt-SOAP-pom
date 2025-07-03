#!/bin/bash

# Modded NanoGPT Streamlit Demo Launcher
echo "🚀 Starting Modded NanoGPT Interactive Demo..."

# Set up tiktoken cache directory
export TIKTOKEN_CACHE_DIR=".tiktoken_cache"
mkdir -p .tiktoken_cache

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Launch Streamlit app
echo "🌐 Launching demo at http://localhost:8501"
streamlit run streamlit_demo.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.serverAddress localhost \
    --server.enableCORS false \
    --server.enableXsrfProtection false 