# AcceptableAds Crawler

This repository contains a web crawler implementation for collecting and analyzing acceptable ads data. The crawler is built on top of DuckDuckGo's Tracker Radar Collector with several custom modifications.

## Features

- Ad detection and collection
- Fingerprinting detection
- Screen recording
- Cookie and request tracking
- CMP (Consent Management Platform) interaction
- Screenshot capture

## Implementation Details


### Data release
We plan to provide the data on request to research labs who could benefit from the high quality labelled dataset. Please send a request email to ritik.r@nyu.edu with the subject `AccAds Dataset Request`. 

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

```sh
python3 wrapper_out.py --auth 0
```
`Note:` On some systems, the `npm i` command hangs within the the docker when executed from outside. A hack for it is to attach to the docker manually using `docker attach accads_control` and then execute `cd accads_crawler && npm i` until the command finishes. Later we can detach and run the wrapper_out command as is.


### Authenticated Crawls
On each of the VMs (we used machines with different IPs), we run control and adblock separately i.e. one VM runs `control` and the other runs `adblock`. 
VM-1
```sh
python3 wrapper_out.py --auth 1 --extn control ## For control
```
VM-2
```sh
python3 wrapper_out.py --auth 1 --extn adblock ## For control
```

In order to run the authenticated crawls, we first need to create authenticated profiles. This would need the reviewer to login through Gmail Email and Password. After making the following edits, the browser would open up in GUI mode where the reviewer would need to manually log into their gmail accounts and then close the browser and end the process.
They need to then revert the steps in order to now run the crawls with the authenticated profile.

Edits
- In `crawlConductor.js`, comment `Line 7` and uncomment `Line 8` to use `crawler_auth.js`. Now run the wrapper_out.py commands above. Ensure that you run it on a platform where you have GUI access in order to login.

Revert
- In `crawlConductor.js`, uncomment `Line 7` and comment `Line 8` to use `crawler.js`.
- In `crawler.js`, uncomment `Line 69` and `Line 84` in order to use the temp_session which should contain the authenticated profiles for the crawl.

#### Webpage lists
You can find all crawled URLs, including landing and inner page URLs in the [websites_inner_sites.txt]](https://github.com/Racro/AcceptableAds_PETS/accads_crawler/websites_inner_sites.txt).

## Directory Structure

- `collectors/`: Custom collectors for different data types
- `helpers/`: Helper functions and utilities
- `reporters/`: Output formatting and reporting
- `shell_scripts/`: Utility scripts for running crawls
- `tests/`: Test cases and test data
