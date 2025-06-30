# AcceptableAds Docker Compose Setup

A simplified Docker Compose framework for running the AcceptableAds crawler and processing collected ad images.

## Overview

This setup includes three main services:
- **crawler-control**: Runs crawls without ad blocking
- **crawler-adblock**: Runs crawls with ad blocking
- **processing**: Processes collected ad images (OCR, deduplication, etc.)

## Quick Start

### 1. Setup Environment
```bash
# Copy environment file
cp env.example .env

# Edit configuration as needed
nano .env
```

### 2. Create Required Directories
```bash
mkdir -p data/control data/adblock data/shared config logs
```

### 3. Copy Configuration Files
```bash
# Copy your website list
# Note: websites_1500.txt is used directly from accads_crawler/ directory

# Copy any other configuration files you need
```

### 4. Start Services
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Usage

### Running Crawls
The crawler services use the Docker Hub image `racro/accads:latest`. You can execute crawls by:

```bash
# Run control crawl
docker exec accads-crawler-control python3 wrapper_out.py --auth 0 --extn control

# Run adblock crawl  
docker exec accads-crawler-adblock python3 wrapper_out.py --auth 0 --extn adblock
```

### Processing Ad Images
The processing service specifically targets the `adshots` directories where ad images are stored:

```bash
# Run processing manually
docker exec accads-processing python3 process_images.py

# View processing logs
docker exec accads-processing tail -f /app/logs/processing.log
```

## Data Structure

```
AcceptableAds_PETS/
├── data/
│   ├── control/          # Control crawl results
│   │   ├── adshots/      # Ad images from control crawl
│   │   ├── *.png         # Page screenshots
│   │   └── *.json        # Network information
│   ├── adblock/          # Adblock crawl results
│   │   ├── adshots/      # Ad images from adblock crawl
│   │   ├── *.png         # Page screenshots
│   │   └── *.json        # Network information
│   └── shared/           # Shared data between services
├── logs/                 # Application logs
# Note: Configuration files are used directly from their source locations
```

## Configuration

### Environment Variables
- `CRAWLER_TIMEOUT`: Timeout for crawls (default: 90)
- `CRAWLER_MAX_RETRIES`: Maximum retry attempts (default: 3)
- `OCR_FALLBACK`: Enable Tesseract fallback (default: true)
- `DEBUG`: Enable debug mode (default: false)

### Resource Limits
- Crawler services: 2GB RAM, 1 CPU
- Processing service: 4GB RAM, 2 CPUs

## Development

For development, use the override file:
```bash
# Start with development settings
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

## Troubleshooting

### View Service Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs crawler-control
docker-compose logs processing
```

### Access Container Shell
```bash
# Crawler container
docker exec -it accads-crawler-control bash

# Processing container
docker exec -it accads-processing bash
```

### Check Data Directories
```bash
# List collected data
ls -la data/control/
ls -la data/adblock/

# Check adshots specifically
ls -la data/control/adshots/
ls -la data/adblock/adshots/
```

## Requirements

- Docker and Docker Compose
- At least 8GB RAM available
- 20GB free disk space
- Internet connection for pulling images

## Notes

- The crawler services use the pre-built Docker Hub image `racro/accads:latest`
- Ad images are automatically collected in the `adshots` subdirectories
- The processing service specifically targets these `adshots` directories for OCR and analysis
- All data is stored within the `AcceptableAds_PETS` root directory structure 