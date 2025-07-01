#!/usr/bin/env python3
"""
Script to run the processing container once when data is ready.
"""

import subprocess
import sys
import os

def check_data_ready():
    """Check if there's data to process."""
    control_dir = "./data/control"
    adblock_dir = "./data/adblock"
    
    # Check if directories exist and contain data
    control_has_data = os.path.exists(control_dir) and any(
        os.path.exists(os.path.join(control_dir, d)) 
        for d in os.listdir(control_dir) if os.path.isdir(os.path.join(control_dir, d))
    )
    
    adblock_has_data = os.path.exists(adblock_dir) and any(
        os.path.exists(os.path.join(adblock_dir, d)) 
        for d in os.listdir(adblock_dir) if os.path.isdir(os.path.join(adblock_dir, d))
    )
    
    return control_has_data or adblock_has_data

def run_processing():
    """Run the processing container."""
    print("Starting processing container...")
    
    try:
        # Build and run the processing container
        result = subprocess.run([
            "docker-compose", "up", "--build", "processing"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Processing completed successfully!")
            return True
        else:
            print(f"❌ Processing failed with exit code: {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running processing: {e}")
        return False

def main():
    """Main function."""
    print("Checking if data is ready for processing...")
    
    if not check_data_ready():
        print("❌ No data found to process!")
        print("Please run the crawler first to collect data.")
        sys.exit(1)
    
    print("✅ Data found! Starting processing...")
    
    success = run_processing()
    
    if success:
        print("\n🎉 Processing completed!")
        print("Check the following directories for results:")
        print("- ./image_hashing/ocr_control.json")
        print("- ./image_hashing/ocr_adblock.json")
        print("- ./logs/processing.log")
    else:
        print("\n❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 