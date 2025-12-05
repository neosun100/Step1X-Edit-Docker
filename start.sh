#!/bin/bash

set -e

echo "========================================="
echo "  Step1X-Edit Docker Startup Script"
echo "========================================="

# Check nvidia-docker
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

if ! docker info | grep -q "Runtimes.*nvidia"; then
    echo "❌ Error: nvidia-docker runtime not found. Please install nvidia-docker2."
    exit 1
fi

echo "✓ NVIDIA Docker environment detected"

# Auto-select GPU with least memory usage
echo ""
echo "Selecting GPU with least memory usage..."
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
         sort -t',' -k2 -n | head -1 | cut -d',' -f1)

if [ -z "$GPU_ID" ]; then
    echo "❌ Error: No GPU found"
    exit 1
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID)

echo "✓ Selected GPU $GPU_ID: $GPU_NAME (${GPU_MEM}MB used)"

# Export GPU ID
export NVIDIA_VISIBLE_DEVICES=$GPU_ID

# Load .env if exists
if [ -f .env ]; then
    echo "✓ Loading .env file"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠ Warning: .env file not found, using defaults"
    export PORT=8000
    export GPU_IDLE_TIMEOUT=60
fi

# Start docker-compose
echo ""
echo "Starting Step1X-Edit container..."
docker-compose up -d

# Wait for service to be ready
echo ""
echo "Waiting for service to start..."
sleep 5

# Check if container is running
if docker ps | grep -q step1x-edit; then
    echo ""
    echo "========================================="
    echo "  ✓ Step1X-Edit Started Successfully!"
    echo "========================================="
    echo ""
    echo "Access URLs:"
    echo "  • UI:      http://0.0.0.0:${PORT:-8000}"
    echo "  • API:     http://0.0.0.0:${PORT:-8000}/docs"
    echo "  • Health:  http://0.0.0.0:${PORT:-8000}/health"
    echo ""
    echo "GPU Info:"
    echo "  • GPU ID:  $GPU_ID"
    echo "  • GPU:     $GPU_NAME"
    echo "  • Memory:  ${GPU_MEM}MB used"
    echo ""
    echo "Management:"
    echo "  • Logs:    docker-compose logs -f"
    echo "  • Stop:    docker-compose down"
    echo "  • Restart: docker-compose restart"
    echo ""
else
    echo "❌ Error: Container failed to start"
    echo "Check logs with: docker-compose logs"
    exit 1
fi
