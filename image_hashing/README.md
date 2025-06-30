# Image Hashing and Analysis

This directory contains tools for image analysis, hashing, deduplication, and OCR extraction for ad images collected during the crawling process.

## Overview

The image hashing tools provide image processing capabilities including OCR text extraction, deduplication, and similarity analysis for the collected ad images.

## Key Tools

- **OCR Processing**: `ocr.py` extracts text from ad images using optical character recognition
- **Deduplication**: `deduplicate.py` removes duplicate images using perceptual hashing
- **Similarity Search**: `faiss_compare.py` and `faiss_vector_gen.py` provide fast similarity search using FAISS
- **Image Cleaning**: `cleaning.py` and `detect_white.py` handle image preprocessing and blank image detection

## Features

- Perceptual image hashing for duplicate detection
- Fast similarity search using FAISS indexing
- OCR text extraction with JSON output format
- Image preprocessing and cleaning utilities
- White/blank image detection and filtering

## Integration

- **OCR Extraction**: Automatically performed by the processing service during the Docker workflow
- **Output Files**: OCR results saved as `ocr_control.json` and `ocr_adblock.json` for use in LLM annotation
- **Pipeline Integration**: Used by both the processing service and annotation scripts

## Usage

```bash
# Run image deduplication
python3 deduplicate.py

# Generate FAISS vectors for similarity search
python3 faiss_vector_gen.py

# Compare images using FAISS
python3 faiss_compare.py
```

## Output

- OCR JSON files for each crawl mode (control/adblock)
- Deduplicated image sets
- Image similarity scores and analysis
- FAISS index files for fast similarity search

For detailed crawling and processing instructions, see the main [README.md](../README.md).
