#!/usr/bin/env python3
"""
Script to run the LLM annotation container when data is ready.
"""

import subprocess
import sys
import os
import argparse

def check_data_ready():
    """Check if there's data to process."""
    # Check if OCR files exist (from previous processing)
    control_ocr = "./image_hashing/ocr_control.json"
    adblock_ocr = "./image_hashing/ocr_adblock.json"
    
    # Check if ad images exist
    control_has_images = os.path.exists("./data/control/adshots") and any(
        f.endswith('.png') for f in os.listdir("./data/control/adshots")
    )
    
    adblock_has_images = os.path.exists("./data/adblock/adshots") and any(
        f.endswith('.png') for f in os.listdir("./data/adblock/adshots")
    )
    
    return (os.path.exists(control_ocr) or os.path.exists(adblock_ocr)) and (control_has_images or adblock_has_images)

def get_api_key():
    """Get OpenAI API key from environment variable or command line argument."""
    parser = argparse.ArgumentParser(description='LLM annotation for ad analysis')
    parser.add_argument('--openai_key', type=str, help='OpenAI API key')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key (alternative format)')
    args, _ = parser.parse_known_args()
    
    # Try command line argument first
    api_key = args.openai_key or args.openai_key
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please provide it via --openai_key argument or set the OPENAI_KEY environment variable.")
    
    return api_key

def run_llm_annotation(api_key=None):
    """Run the LLM annotation container."""
    print("Starting LLM annotation container...")
    
    try:
        # Build the command
        cmd = ["docker-compose", "run", "--rm", "processing", "python3", "/app/processing_scripts/llm_annotation.py"]
        
        # If API key is provided, pass it as environment variable
        if api_key:
            cmd = ["docker-compose", "run", "--rm", "-e", f"OPENAI_KEY={api_key}", "processing", "python3", "/app/processing_scripts/llm_annotation.py"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ LLM annotation completed successfully!")
            return True
        else:
            print(f"❌ LLM annotation failed with exit code: {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running LLM annotation: {e}")
        return False

def main():
    """Main function."""
    print("Checking if data is ready for LLM annotation...")
    
    if not check_data_ready():
        print("❌ No data found to annotate!")
        print("Please run the processing first to generate OCR data.")
        print("Run: python3 run_processing.py")
        sys.exit(1)
    
    print("✅ Data found! Starting LLM annotation...")
    
    # Try to get API key, but don't fail if not provided (will use .env file)
    api_key = None
    try:
        api_key = get_api_key()
    except ValueError:
        print("⚠️  No API key provided via command line or environment variable.")
        print("Will use API key from .env file if available.")
    
    success = run_llm_annotation(api_key)
    
    if success:
        print("\n🎉 LLM annotation completed!")
        print("Check the following files for results:")
        print("- ./processing_scripts/llm_annotation_dict.json")
        print("- ./processing_scripts/llm_annotation_explanations.json")
    else:
        print("\n❌ LLM annotation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 