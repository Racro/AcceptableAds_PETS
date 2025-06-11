# Artifact Appendix

Paper title: Sheep’s clothing, wolfish impact: Automated detection and evaluation of problematic 'allowed' advertisements

Artifacts HotCRP Id: 29

Requested Badge: **Available**, **Functional**

## Description
The artifact provides the crawling code for scraping [online ads](./accads_crawler), cleaning redundant screenshots and [deduplicating ads](./image_hashing).
Scripts to collect LLM annotations for ads and script to calculate agreement scores are also [provided](./processing_scripts/). 

### Security/Privacy Issues and Ethical Concerns (All badges)
The code does not pose any security/privacy issues or ethical concerns. Crawling the entire website dataset might cause certain ethical concerns but for the sake of reproducibility, the reviewers can crawl a small subset (10 websites) and collect ads.

## Basic Requirements (Only for Functional and Reproduced badges)
The artifacts are good enough to run on a laptop.
For LLM annotations, the reviewers would require an OPENAI API key which they can set in their local terminal environment.
For running OCR, we used Google VISION API. Hence the reviewers will need to set that up as well. That part can be commented out as well.

### Hardware Requirements
We used 2 VMs with different IPs for the Authenticated Crawls so that similar IPs do not affect ad content. They are not required to check the functionality of the code as only one instance can be run on the local machine (considering it as one of the VMs)

### Software Requirements
Docker container is required to run unauthenticated crawls.
For authenticated crawls, the user needs to create authenticated profiles which are crucial for capturing ads. Details are provided in the README.


### Estimated Time and Storage Consumption
The crawls might take 2 days to finish over the entire website dataset but to conduct it over a small subset of 10 websites, it should take less than 10 minutes.
The processing scripts are fairly quick too.

## Environment 
Having docker set up and running would be crucial. 
Every folder is provided with a requirements.txt which could be loaded in a python virtual environment. 

### Accessibility (All badges)
The artifact is hosted on a public [GitHub repository](https://github.com/Racro/AcceptableAds_PETS). 
The master branch would contain the most updated version. The current commit-id is 
The dataset is available at 

### Set up the environment (Only for Functional and Reproduced badges)

Clone the Repo
```bash
git clone https://github.com/Racro/AcceptableAds_PETS.git
```

Run the crawler and collect ads (Remember to trim the websites_inner_sites.txt to crawl only few sites)
```bash
cd accads_crawler && python3 wrapper_out.py --auth 0 ## Unathenticated crawls 
cd accads_crawler && python3 wrapper_out.py --auth 1 --extn control/adblock ## Athenticated crawls 
```

Run the deduplication script on a sample dataset
```bash
cd image_hashing && python3 -m venv dedup
pip install -r requirements.txt
python3 deduplicate.py
```

For the Agreement scores
```bash
cd processing_scripts && python3 -m venv scripts
pip install -r requirements.txt
python3 iaa_jaccard.py
```

For the LLM Annotations
```bash
python3 llm_annotation.py
```

### Testing the Environment (Only for Functional and Reproduced badges)
The scripts running should be a test for the environment functionality as well.

## Artifact Evaluation (Only for Functional and Reproduced badges)
Please refer to the order of the steps above to execute different codebases.
We can only validate LLM annotations (that too also to an extent since there is a monetary cost) and the inter annotator agreements on the provided dataset.

### Main Results and Claims
#### Main Result 1: LLM Agreement scores (Table 3)
LLM annotations can be obtained and then matched with the ground_truth to obtain agreement scores.

## Limitations (Only for Functional and Reproduced badges)
All the dataset using expert labellers couldn't be reproduced using the provided artifact.

## Notes on Reusability (Only for Functional and Reproduced badges)
We believe this artifact can be expanded to collect a larger pool of ads and extend the viability of LLMs to automate the annotation process.