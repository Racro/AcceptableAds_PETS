# AcceptableAds Crawler

This repository contains a web crawler implementation for collecting and analyzing acceptable ads data. The crawler is built on top of DuckDuckGo's Tracker Radar Collector with several custom modifications.

## Features

- Ad detection and collection
- Fingerprinting detection
- Screen recording
- Cookie and request tracking
- CMP (Consent Management Platform) interaction
- Screenshot capture

## Installation

1. Clone this repo:
    ```sh
    git clone https://github.com/Racro/AcceptableAds_PETS.git
    cd accads_crawler
    ```

2. Install the required npm packages:
    ```sh
    npm i
    ```

## Usage

### Unauthenticated Crawl

By default, the crawler is designed to run inside a Docker container for ease of setup and reproducibility. If you want to use the pre-built Docker image, you can simply pull and run it as shown below. Alternatively, if you prefer to set up your own Docker environment, you can do so using the provided Docker commands.

**Option 1: Using the Pre-built Docker Image**

1. Pull the latest Docker image:
    ```sh
    docker pull racro/accads:latest
    ```

2. Run the container and start the crawl:
    ```sh
    python3 wrapper_out.py --auth 0
    ```

**Option 2: Setting Up Your Own Docker Environment**

If you want to build and run the Docker container yourself:

1. Build the Docker image:
    ```sh
    docker build -t accads .
    ```

2. Run the container:
    ```sh
    docker run -it --name accads_control accads
    cd accads_crawler
    npm i
    python3 wrapper_out.py --auth 0
    ```

**Note:**  
On some systems, the `npm i` command may hang when executed from outside the Docker container. If this happens, run `npm i` from inside the container as shown above.

### Authenticated Crawls

For authenticated crawls, you need to create authenticated browser profiles by logging in with Gmail credentials. This process requires manual login via a GUI browser session. The authenticated profiles are then used for crawling. Typically, you run the `control` and `adblock` crawls on separate VMs (each with a different IP address).

**Step 1: Prepare Authenticated Profiles**

1. Edit `crawlConductor.js`:
    - Comment out **Line 7** and uncomment **Line 8** to use `crawler_auth.js` instead of `crawler.js`.
2. Start the container (with GUI access) and run:
    ```sh
    python3 wrapper_out.py --auth 1 --extn control  # or --extn adblock
    ```
3. When the browser opens, log in to your Gmail account manually. Once logged in, close the browser and stop the process.

**Step 2: Revert Edits for Crawling**

1. In `crawlConductor.js`, revert the changes:
    - Uncomment **Line 7** and comment **Line 8** to use `crawler.js`.
2. In `crawler.js`, uncomment **Line 69** and **Line 84** to enable the use of the `temp_session` containing the authenticated profiles.

**Step 3: Run Authenticated Crawls**

- On VM-1 (for control):
    ```sh
    python3 wrapper_out.py --auth 1 --extn control
    ```
- On VM-2 (for adblock):
    ```sh
    python3 wrapper_out.py --auth 1 --extn adblock
    ```

**Note:**
- Ensure you have GUI access for the authentication step.
- Each VM should run only one extension type (`control` or `adblock`).

#### Webpage lists
You can find all crawled URLs, including landing and inner page URLs in the [websites_inner_sites.txt]](https://github.com/Racro/AcceptableAds_PETS/accads_crawler/websites_inner_sites.txt).

### Data Storage and Output Structure

After running a crawl (either control or adblock), the collected data is stored in separate folders named `control` and `adblock`, corresponding to the type of crawl performed. Each of these folders contains various files and subdirectories with different types of information:

- **PNG Images (outside subfolders):**
  - These are page screenshots captured during the crawl. Each image corresponds to a specific page state or adshot.

- **JSON Files (outside subfolders):**
  - These files contain network information, such as requests and responses observed during the crawl session.

- **adData/ Directory:**
  - Contains JSON files with detailed network information specifically about ads detected on the crawled pages.

- **adshots/ Directory:**
  - Stores images (typically PNGs) of detected ads (adshots) that were found on the pages during the crawl.

The same structure is present in both the `control` and `adblock` folders, allowing you to compare results between the two modes easily.

## Directory Structure

- `collectors/`: Custom collectors for different data types
- `helpers/`: Helper functions and utilities
- `reporters/`: Output formatting and reporting
- `shell_scripts/`: Utility scripts for running crawls
- `tests/`: Test cases and test data
