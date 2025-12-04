#!/bin/bash
# Quick start script for setting up and running the project

set -e

echo "🚀 Embedding Model Fine-tuning - Quick Start"
echo "==========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python version: $PYTHON_VERSION"
echo ""

# Ask user which method they want to use
echo "Please select installation method:"
echo "1) pip (Virtual Environment)"
echo "2) Poetry"
echo "3) Docker"
echo "4) Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "📦 Setting up with pip and virtual environment..."
        
        # Create virtual environment if it doesn't exist
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            echo "✅ Virtual environment created"
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        echo "✅ Virtual environment activated"
        
        # Install dependencies
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "✅ Dependencies installed"
        
        echo ""
        echo "🎉 Setup complete!"
        echo "To run the training script:"
        echo "  source venv/bin/activate"
        echo "  python src/main.py"
        ;;
    
    2)
        echo ""
        echo "📦 Setting up with Poetry..."
        
        # Check if Poetry is installed
        if ! command -v poetry &> /dev/null; then
            echo "❌ Poetry is not installed."
            echo "Install it with: curl -sSL https://install.python-poetry.org | python3 -"
            exit 1
        fi
        
        # Install dependencies
        poetry install
        echo "✅ Dependencies installed"
        
        echo ""
        echo "🎉 Setup complete!"
        echo "To run the training script:"
        echo "  poetry run python src/main.py"
        ;;
    
    3)
        echo ""
        echo "🐳 Setting up with Docker..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker is not installed. Please install Docker first."
            exit 1
        fi
        
        # Build Docker image
        echo "Building Docker image..."
        docker build -t embedding-finetuning .
        echo "✅ Docker image built"
        
        echo ""
        echo "🎉 Setup complete!"
        echo "To run the training script:"
        echo "  docker run -v \$(pwd)/finetuned_finance_model:/app/finetuned_finance_model embedding-finetuning"
        echo ""
        echo "Or use Docker Compose:"
        echo "  docker-compose up"
        ;;
    
    4)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "📝 Note: The training script uses dummy data by default."
echo "   To use your own data, prepare a CSV file with 'term' and 'definition' columns"
echo "   and update the data loading section in src/main.py"
echo ""
echo "🔍 Run 'python verify_setup.py' to verify your setup"
