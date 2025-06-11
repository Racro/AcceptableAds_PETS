# Image Hashing and Analysis

This directory contains tools for image analysis, hashing, and deduplication of ad images.

## Tools Overview

- `cleaning.py`: Image cleaning and preprocessing utilities
- `deduplicate.py`: Image deduplication using perceptual hashing
- `faiss_compare.py`: Fast similarity search using FAISS
- `faiss_vector_gen.py`: Generate FAISS vectors for image comparison
- `ocr.py`: Optical Character Recognition for ad text extraction
- `detect_white.py`: Detection of white/blank images

## Features

- Perceptual image hashing
- Fast similarity search using FAISS
- OCR for text extraction
- Image deduplication
- White/blank image detection

## Usage

1. Install dependencies:
```bash
python3.10 -m venv dedup
pip install -r requirements.txt
pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu
```

2. Run deduplication script:
```bash
python deduplicate.py
```

## Output

The tools generate:
- Deduplicated image sets
- Image similarity scores
- Extracted text from images
- FAISS index files for fast similarity search
