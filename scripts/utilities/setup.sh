#!/bin/bash

# NeuronMap Setup Script
# ======================

set -e  # Exit on any error

echo "🚀 Setting up NeuronMap..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python $PYTHON_VERSION is compatible"
else
    echo "❌ Python 3.8+ required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "🔨 Installing NeuronMap in development mode..."
pip install -e .

# Check if Ollama is installed
echo "🔍 Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if Ollama is running
    if ollama list &> /dev/null; then
        echo "✓ Ollama is running"
    else
        echo "⚠️ Ollama is installed but not running. Start it with: ollama serve"
    fi
else
    echo "❌ Ollama not found. Please install from: https://ollama.ai/"
    echo "   After installation, pull a model: ollama pull deepseek-r1:32b"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{raw,processed,outputs}

# Run validation
echo "✅ Running system validation..."
python main.py validate

echo ""
echo "🎉 NeuronMap setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. If Ollama isn't running: ollama serve"
echo "3. Pull a model: ollama pull deepseek-r1:32b"
echo "4. Generate questions: python main.py generate --config dev"
echo "5. Extract activations: python main.py extract --list-layers"
echo "6. Create visualizations: python main.py visualize"
echo ""
echo "For help: python main.py --help"
