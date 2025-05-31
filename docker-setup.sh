#!/bin/bash

# Mobile Crashes RCA Agent - Docker Setup Script
# This script sets up the local development environment with Langfuse observability

set -e

echo "🚀 Mobile Crashes RCA Agent - Docker Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed (try both docker-compose and docker compose)
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Use docker compose if available, fallback to docker-compose
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.template .env
    echo "✅ Created .env file. Please edit it with your configuration:"
    echo "   - Add your OPENAI_API_KEY"
    echo "   - Configure Langfuse keys (after setup)"
    echo ""
    echo "❗ Please edit .env file before continuing."
    echo "Press any key to continue once you've updated .env..."
    read -n 1 -s
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/cache data/outputs data/embeddings logs

# Pull latest images
echo "📦 Pulling Docker images..."
$COMPOSE_CMD pull

# Start the services
echo "🔄 Starting services..."
$COMPOSE_CMD up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check if Langfuse is running
echo "🔍 Checking Langfuse status..."
if curl -f http://localhost:3000/api/public/health > /dev/null 2>&1; then
    echo "✅ Langfuse is running at http://localhost:3000"
else
    echo "⏳ Langfuse is still starting up..."
    echo "   You can check the status with: $COMPOSE_CMD logs langfuse-server"
fi

# Show status
echo ""
echo "📊 Service Status:"
echo "=================="
$COMPOSE_CMD ps

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Open Langfuse at http://localhost:3000"
echo "2. Create a new project in Langfuse"
echo "3. Copy the public and secret keys to your .env file"
echo "4. Run: docker-compose restart mobile-rca-agent"
echo ""
echo "Useful commands:"
echo "- View logs: $COMPOSE_CMD logs -f [service-name]"
echo "- Stop services: $COMPOSE_CMD down"
echo "- Restart: $COMPOSE_CMD restart"
echo "- Rebuild: $COMPOSE_CMD up --build"
echo ""
echo "Access points:"
echo "- Langfuse UI: http://localhost:3000"
echo "- Agent API: http://localhost:8000 (if running)"
echo "- PostgreSQL: localhost:5432 (langfuse/langfuse)" 