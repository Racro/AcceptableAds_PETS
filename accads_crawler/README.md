# AcceptableAds Crawler

This directory contains the web crawler implementation for collecting and analyzing acceptable ads data. The crawler is built on top of DuckDuckGo's Tracker Radar Collector with several custom modifications.

## Features

- Ad detection and collection
- Fingerprinting detection
- Screen recording
- Cookie and request tracking
- CMP (Consent Management Platform) interaction
- Screenshot capture

## Directory Structure

- `collectors/`: Custom collectors for different data types
  - `AdCollector.js`: Collects ad-related data and screenshots
  - `APICallCollector.js`: Tracks API calls and network requests
  - `BaseCollector.js`: Base class for all collectors
  - `CMPCollector.js`: Handles consent management platform interactions
  - `FingerprintCollector.js`: Detects browser fingerprinting attempts
  - `NetworkCollector.js`: Collects network traffic data
  - `ScreenshotCollector.js`: Captures page screenshots
  - `VideoCollector.js`: Records screen activity
- `helpers/`: Helper functions and utilities
  - `collectorsList.js`: Manages collector registration
  - `deferred.js`: Promise-based utilities
  - `dismissDialog.js`: Handles dialog dismissal
  - `headers.js`: HTTP header utilities
  - `initiators.js`: Request initiator tracking
  - `logger.js`: Logging utilities
  - `network.js`: Network-related helpers
  - `screenshot.js`: Screenshot utilities
  - `video.js`: Video recording helpers
- `reporters/`: Output formatting and reporting
  - `BaseReporter.js`: Base reporter class
  - `ClickhouseReporter.js`: ClickHouse database reporting
  - `CLIReporter.js`: Command-line interface reporting
  - `JSONReporter.js`: JSON format reporting
- `shell_scripts/`: Utility scripts for running crawls
- `tests/`: Test cases and test data
- `extn_src/`: Browser extension source code

## Key Files

- `crawler.js`: Main crawler implementation for unauthenticated crawls
- `crawler_auth.js`: Authenticated crawler implementation
- `crawlConductor.js`: Orchestrates the crawling process
- `wrapper_out.py`: Python wrapper for running crawls
- `websites_1500.txt`: List of websites to crawl
- `websites_inner_sites.txt`: Complete list of crawled URLs including inner pages

## Technical Implementation

The crawler uses a modular architecture with collectors that gather different types of data:

1. **Base Collector**: Provides common functionality for all collectors
2. **Specialized Collectors**: Each collector focuses on a specific data type
3. **Reporters**: Format and output the collected data
4. **Helpers**: Provide utility functions and common operations

The crawler supports both authenticated and unauthenticated modes, with the main difference being the use of browser profiles for authentication.

## Browser Extensions

The `extn_src/` directory contains three versions of the AdBlock Plus extension:
- `adblock/`: Original version
- `adblock_v2/`: Manifest Version 2
- `adblock_v3/`: Manifest Version 3

Each extension version includes localization files, icons, and the necessary JavaScript for ad blocking functionality.

## Configuration

The crawler can be configured through various files:
- `crawlConductor.js`: Main crawling configuration
- `crawler.js`/`crawler_auth.js`: Crawler-specific settings
- Environment variables for Docker deployments
- Extension configuration files in `extn_src/`

For detailed crawling instructions, see the main [README.md](../README.md).
