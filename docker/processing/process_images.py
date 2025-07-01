#!/usr/bin/env python3
"""
Simple image processing script for AcceptableAds crawler output.
Processes ad images from control and adblock adshots directories.
"""

import os
import sys
import glob
import time
import json
from pathlib import Path
import logging

# Add parent directory to path to import image_hashing modules
sys.path.append('/app/image_hashing')

# Import OCR with fallback for other modules
try:
    from image_hashing.ocr import detect_text
except ImportError as e:
    logging.error(f"Could not import OCR module: {e}")
    detect_text = None

# Try to import other modules, but don't fail if they're missing
try:
    from image_hashing.deduplicate import deduplicate_images
except ImportError:
    deduplicate_images = None
    logging.warning("Deduplication module not available")

try:
    from image_hashing.detect_white import detect_white_images
except ImportError:
    detect_white_images = None
    logging.warning("White detection module not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/processing.log'),
        logging.StreamHandler()
    ]
)

def save_ocr_data(ocr_data, extension_type):
    """Save OCR data to JSON file organized by extension type."""
    output_dir = "/app/image_hashing"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"ocr_{extension_type}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(ocr_data, f, indent=2)
        logging.info(f"Saved OCR data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving OCR data to {output_file}: {e}")

def process_adshots_directory(directory_path, extension_type):
    """Process all ad images in the adshots directory and save OCR data."""
    adshots_path = os.path.join(directory_path, "adshots")
    
    if not os.path.exists(adshots_path):
        logging.warning(f"Adshots directory {adshots_path} does not exist")
        return {}
    
    logging.info(f"Processing adshots directory: {adshots_path}")
    
    # Find all PNG images in adshots
    image_files = glob.glob(os.path.join(adshots_path, "*.png"))
    logging.info(f"Found {len(image_files)} ad images in {adshots_path}")
    
    ocr_data = {}
    
    # Process each ad image
    for image_path in image_files:
        try:
            filename = os.path.basename(image_path)
            logging.info(f"Processing ad image: {filename}")
            
            # Extract text using OCR if available
            if detect_text:
                text = detect_text(image_path)
                if text:
                    ocr_data[filename] = text
                    logging.info(f"Extracted text from {filename}: {text[:100]}...")
                else:
                    ocr_data[filename] = ""
                    logging.info(f"No text extracted from {filename}")
            else:
                logging.warning("OCR not available, skipping text extraction")
                ocr_data[filename] = ""
            
            # You can add more processing steps here when dependencies are available
            # - Image deduplication
            # - White image detection
            # - Feature extraction
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            ocr_data[os.path.basename(image_path)] = ""
    
    return ocr_data

def process_directory(directory_path, extension_type):
    """Process only adshots directory and save OCR data."""
    if not os.path.exists(directory_path):
        logging.warning(f"Directory {directory_path} does not exist")
        return
    
    logging.info(f"Processing directory: {directory_path}")
    
    # Process only adshots directory
    adshots_ocr_data = process_adshots_directory(directory_path, extension_type)
    
    # Save adshots OCR data
    if adshots_ocr_data:
        save_ocr_data(adshots_ocr_data, extension_type)

def main():
    """Main processing function that runs once and exits."""
    logging.info("Starting image processing (single run)...")
    
    try:
        # Process control directory
        control_dir = "/app/data/control"
        process_directory(control_dir, "control")
        
        # Process adblock directory
        adblock_dir = "/app/data/adblock"
        process_directory(adblock_dir, "adblock")
        
        logging.info("Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 