#!/bin/bash
# Start script for qEEG Explainer Video Generator
# Uses Python 3.10 (required for Kokoro TTS)

PYTHON="/opt/homebrew/bin/python3.10"

# Check if Python 3.10 exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: Python 3.10 not found at $PYTHON"
    echo "Install with: brew install python@3.10"
    exit 1
fi

# Check for required packages
$PYTHON -c "import streamlit, kokoro, anthropic, openai, replicate" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    $PYTHON -m pip install -r requirements.txt
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run the app
echo "Starting qEEG Explainer Video Generator..."
echo "Opening http://localhost:8501"
$PYTHON -m streamlit run app.py "$@"
