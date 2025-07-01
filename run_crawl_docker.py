#!/usr/bin/env python3
"""
Docker-adapted version of wrapper_out.py for Docker Compose setup.
"""

import subprocess
import os
import time
import argparse
import sys

def create_data_directories():
    """Create data directories if they don't exist."""
    dirs = ["./data/control", "./data/adblock", "./data/shared"]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def run_single_url_in_container(container_name, url, crawler_type):
    """Run a single URL in the specified container."""
    
    # Kill any existing crawl processes first
    try:
        subprocess.run([
            "docker", "exec", container_name, 
            "pkill", "-f", "node.*crawl-cli"
        ], capture_output=True)
        time.sleep(2)
    except:
        pass
    
    # Create the crawl command
    if crawler_type == "control":
        cmd = [
            "docker", "exec", container_name,
            "bash", "-c", 
            f"cd /home/chromiumuser/AcceptableAds_PETS/accads_crawler && npm run crawl -- -u '{url}' -o ./control -v -f -d ads --reporters cli,file -l ./control/ --autoconsent-action optIn"
        ]
    else:  # adblock
        cmd = [
            "docker", "exec", container_name,
            "bash", "-c", 
            f"cd /home/chromiumuser/AcceptableAds_PETS/accads_crawler && npm run crawl -- -u '{url}' -o ./adblock -v -f -d ads --reporters cli,file -l ./adblock/ --autoconsent-action optOut"
        ]
    
    print(f"Running {crawler_type} crawl for: {url}")
    
    try:
        # Run with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {crawler_type} crawl completed successfully!")
            return True
        else:
            print(f"❌ {crawler_type} crawl failed with exit code: {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr[-200:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {crawler_type} crawl timed out after 3 minutes")
        # Kill the process
        try:
            subprocess.run([
                "docker", "exec", container_name, 
                "pkill", "-f", "node.*crawl-cli"
            ], capture_output=True)
        except:
            pass
        return False
    except Exception as e:
        print(f"❌ {crawler_type} crawl error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run crawls using Docker Compose containers')
    parser.add_argument('--extn', type=str, default='control', choices=['control', 'adblock'], 
                       help='Extension type: control or adblock')
    parser.add_argument('--urls-file', type=str, default='accads_crawler/websites_1500.txt',
                       help='File containing URLs to crawl')
    parser.add_argument('--num-urls', type=int, default=5,
                       help='Number of URLs to crawl')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per URL in seconds')
    
    args = parser.parse_args()
    
    # Create data directories
    create_data_directories()
    
    # Read URLs from file
    try:
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()][:args.num_urls]
    except FileNotFoundError:
        print(f"Error: URLs file '{args.urls_file}' not found")
        sys.exit(1)
    
    if not urls:
        print("Error: No URLs found in file")
        sys.exit(1)
    
    # Set container name based on extension
    container_name = f"accads-crawler-{args.extn}"
    
    print(f"Running {args.extn} crawl for {len(urls)} URLs:")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")
    
    # Process URLs one by one
    successful_crawls = 0
    failed_crawls = 0
    start_time = time.time()
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"Processing URL {i}/{len(urls)}: {url}")
        print(f"Time elapsed: {time.time() - start_time:.1f}s")
        print(f"{'='*60}")
        
        success = run_single_url_in_container(container_name, url, args.extn)
        
        if success:
            successful_crawls += 1
            print(f"✅ URL {i} completed successfully!")
        else:
            failed_crawls += 1
            print(f"❌ URL {i} failed!")
        
        # Wait between crawls
        if i < len(urls):
            print("Waiting 3 seconds before next crawl...")
            time.sleep(3)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Extension: {args.extn}")
    print(f"Total URLs: {len(urls)}")
    print(f"Successful: {successful_crawls}")
    print(f"Failed: {failed_crawls}")
    print(f"Success rate: {(successful_crawls/len(urls)*100):.1f}%")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per URL: {total_time/len(urls):.1f}s")
    print(f"{'='*60}")
    
    if successful_crawls > 0:
        print(f"\n✅ {args.extn} crawl completed!")
        print(f"Data saved to: ./data/{args.extn}/")
    else:
        print(f"\n❌ All {args.extn} crawls failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 