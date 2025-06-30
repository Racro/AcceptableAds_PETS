# Processing Scripts

This directory contains scripts for processing and analyzing the collected ad data from the AcceptableAds pipeline.

## Overview

The processing scripts handle the post-crawl analysis of ad images, including LLM-based annotation for ad categorization.

## Key Components

- **LLM Annotation**: `llm_annotation.py` generates automated ad category annotations using OpenAI's API
- **Agreement Analysis**: `iaa_jaccard.py` computes inter-annotator agreement scores for multiple annotation sources

## Workflow

1. **LLM Annotation**: Run the annotation script to categorize ads
2. **Analysis**: Use agreement analysis tools to evaluate annotation quality

## Usage

```bash
# Run LLM annotation with OpenAI API
python3 llm_annotation.py --openai_key YOUR_OPENAI_KEY

# Compute inter-annotator agreement (if applicable)
python3 iaa_jaccard.py
```

## Dependencies

- Python 3.8+
- OpenAI API key for LLM annotation
- Required packages from `../image_hashing/requirements.txt`

## Output

- LLM annotation results (JSON and CSV formats)
- Agreement scores and analysis reports
- All outputs saved in the `processing_scripts/` directory

For detailed crawling and processing instructions, see the main [README.md](../README.md).
