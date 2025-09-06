#!/bin/bash

# Test script for Qdrant persistence
echo "🔍 Testing Qdrant Vector Database Persistence"
echo "=============================================="

# Check if services are running
echo "1. Checking services status..."
docker-compose ps

echo ""
echo "2. Checking storage info..."
curl -s http://localhost:8000/storage/info | python -m json.tool

echo ""
echo "3. Listing qdrant_data directory contents..."
if [ -d "./qdrant_data" ]; then
    echo "📁 qdrant_data directory exists:"
    ls -la ./qdrant_data/
    
    if [ -d "./qdrant_data/collections" ]; then
        echo ""
        echo "📚 Collections:"
        ls -la ./qdrant_data/collections/
    fi
    
    if [ -d "./qdrant_data/snapshots" ]; then
        echo ""
        echo "📸 Snapshots:"
        ls -la ./qdrant_data/snapshots/
    fi
else
    echo "❌ qdrant_data directory does not exist yet"
fi

echo ""
echo "4. Testing persistence by restarting containers..."
echo "Stopping containers..."
docker-compose down

echo "Starting containers again..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "5. Checking storage info after restart..."
curl -s http://localhost:8000/storage/info | python -m json.tool

echo ""
echo "✅ Persistence test complete!"
echo "If collections and data persist after restart, persistence is working correctly."
