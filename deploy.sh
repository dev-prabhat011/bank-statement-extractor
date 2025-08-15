#!/bin/bash

echo "🚀 Bank Statement Extractor - Deployment Script"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Download from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads outputs logs ssl

# Set proper permissions
echo "🔐 Setting permissions..."
chmod 755 uploads outputs logs

# Build and start the application
echo "🔨 Building and starting the application..."
docker-compose up --build -d

# Wait for the application to start
echo "⏳ Waiting for the application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:5000/health &> /dev/null; then
    echo "✅ Application is running successfully!"
    echo ""
    echo "🌐 Access your application at:"
    echo "   Local: http://localhost:5000"
    echo "   Network: http://$(hostname -I | awk '{print $1}'):5000"
    echo ""
    echo "📁 Uploads directory: ./uploads"
    echo "📁 Outputs directory: ./outputs"
    echo "📁 Logs directory: ./logs"
    echo ""
    echo "🔧 Management commands:"
    echo "   View logs: docker-compose logs -f"
    echo "   Stop: docker-compose down"
    echo "   Restart: docker-compose restart"
    echo "   Update: docker-compose pull && docker-compose up -d"
else
    echo "❌ Application failed to start. Check logs with:"
    echo "   docker-compose logs"
    exit 1
fi
