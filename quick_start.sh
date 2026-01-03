#!/bin/bash
# Quick start script for Docker-based embedding model fine-tuning

set -e

echo "🚀 Embedding Model Fine-tuning - Docker Quick Start"
echo "===================================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker is installed"

# Check if Docker Compose is available
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    echo "✅ Docker Compose is available (plugin)"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "✅ Docker Compose is available (standalone)"
else
    echo "❌ Docker Compose is not installed."
    echo "Please install Docker Compose plugin"
    exit 1
fi

# Check for NVIDIA Docker (optional but recommended for GPU)
if command -v nvidia-smi &> /dev/null && docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "✅ NVIDIA Docker runtime is available (GPU support enabled)"
    GPU_AVAILABLE=true
else
    echo "⚠️  NVIDIA Docker runtime not detected (GPU support disabled)"
    echo "   Install NVIDIA Container Toolkit for GPU support:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    GPU_AVAILABLE=false
fi

echo ""
echo "Please select an option:"
echo "1) Build and start Docker containers (recommended)"
echo "2) Start existing containers"
echo "3) Stop containers"
echo "4) Rebuild containers (clean build)"
echo "5) Open VSCode Dev Container"
echo "6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "🐳 Building and starting Docker containers..."
        $COMPOSE_CMD up -d --build
        echo ""
        echo "✅ Containers are running!"
        echo ""
        echo "📊 Access Jupyter Lab at: http://localhost:8888"
        echo ""
        echo "To view logs:"
        echo "  $COMPOSE_CMD logs -f"
        echo ""
        echo "To stop containers:"
        echo "  $COMPOSE_CMD down"
        ;;
    
    2)
        echo ""
        echo "🐳 Starting Docker containers..."
        $COMPOSE_CMD up -d
        echo ""
        echo "✅ Containers are running!"
        echo ""
        echo "📊 Access Jupyter Lab at: http://localhost:8888"
        ;;
    
    3)
        echo ""
        echo "🛑 Stopping Docker containers..."
        $COMPOSE_CMD down
        echo "✅ Containers stopped"
        ;;
    
    4)
        echo ""
        echo "🔨 Rebuilding containers (this may take several minutes)..."
        $COMPOSE_CMD down
        $COMPOSE_CMD build --no-cache
        $COMPOSE_CMD up -d
        echo ""
        echo "✅ Containers rebuilt and running!"
        echo ""
        echo "📊 Access Jupyter Lab at: http://localhost:8888"
        ;;
    
    5)
        echo ""
        echo "📝 To use VSCode Dev Container:"
        echo "1. Install 'Dev Containers' extension in VSCode"
        echo "2. Open this folder in VSCode"
        echo "3. Press F1 and select 'Dev Containers: Reopen in Container'"
        echo ""
        echo "The container will be built automatically and GPU support will be enabled."
        ;;
    
    6)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
if [ "$GPU_AVAILABLE" = true ]; then
    echo "🎮 GPU Support: Enabled"
    echo "   Your RTX 4060 should be available inside the container"
else
    echo "⚠️  GPU Support: Disabled"
    echo "   Training will run on CPU (slower)"
fi

echo ""
echo "📂 Data folder: ./data (mounted in container)"
echo "📁 Models will be saved to: ./finetuned_finance_model"
echo "📓 Main notebook: Embedding_model_fine_tuning_test.ipynb"
echo ""
echo "For more details, see DOCKER_SETUP.md"
