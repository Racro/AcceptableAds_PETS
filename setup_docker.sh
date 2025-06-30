#!/bin/bash

echo "Setting up AcceptableAds Docker Compose environment..."

# Create required directories
echo "Creating directories..."
mkdir -p data/control data/adblock data/shared logs

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp env.example .env
    echo "Please edit .env file with your configuration"
else
    echo ".env file already exists"
fi

# Check for websites file
echo "Checking for websites file..."
if [ ! -f accads_crawler/websites_1500.txt ]; then
    echo "Warning: accads_crawler/websites_1500.txt not found"
fi

# Set permissions
echo "Setting permissions..."
chmod 755 data/control data/adblock data/shared logs

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file if needed: nano .env"
echo "2. Start services: docker-compose up -d"
echo "3. Run crawls:"
echo "   docker exec accads-crawler-control python3 wrapper_out.py --auth 0 --extn control"
echo "   docker exec accads-crawler-adblock python3 wrapper_out.py --auth 0 --extn adblock"
echo "4. Process images: docker exec accads-processing python3 process_images.py" 