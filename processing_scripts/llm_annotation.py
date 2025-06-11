import base64
from openai import OpenAI
import os
import pandas as pd
import json
import requests
import sys
import re
import time
# # Initialize OpenAI API with your key
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_KEY environment variable.")

# # Initialize OpenAI client
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
    # return image_file
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def check_dp(strr):
    # check if an alphanumeric character is followed by a '...' or ' ...' ('a  ...' should not be considered)
    # return True if found, False otherwise
    pattern = r"\w(?:(?<! )\.\.\.| \.\.\.)"
    return bool(re.search(pattern, strr))

csv_file = '../sample_data/merged_ground_truth.csv'
df = pd.read_csv(csv_file)

# Extract the first two columns
selected_columns = df.iloc[:, :2]

# Iterate over each row excluding the header
rows = selected_columns.itertuples(index=False, name=None)

# Display the rows
row_list = []
for row in rows:
    row_list.append(row)

all_annot = {}
explanations = {}

for row in row_list:
    filename = row.split('/')[1]
    json_filename = row.split('/')[0]
    image_folder = f'../sample_data/{json_filename}/{filename}'
    if not os.path.exists(image_folder):
        print(json_filename, filename, "DOES NOT EXIST")
        continue

patterns = ['\bsearches', '\bsearch for\b', '\bsearch ad', '\bgosearch', '\bsearchtopics\b', 'suv deals', '\bsuchergebnisse\b', '\bsuchanzeigen\b']

for (key, row) in row_list:
    filename = key.split("'")[1]
    json_filename = row.split("'")[1]

    ocr_text = json.load(open(f'../image_hashing/ocr_{json_filename}.json', 'r'))[filename]

    image_folder = f'../sample_data/{json_filename}/{filename}'
    if not os.path.exists(image_folder):
        all_annot[image_folder] = 'DNA'
        explanations[image_folder] = 'DNA'

    base64_image = encode_image(image_folder)
    # print(ocr_text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
        # body = {
            messages = [
            {
                "role": "user",
                "content": 
                    # f"""Analyze the attached ad and determine if it is problematic or not strictly based on the taxonomy.
                    # image:  "data:image/png;base64,{base64_image}"
                    # """
                    [
                        {  
                            "type": "text",
                            "text": """Analyze the attached ad and determine if it is problematic or non-problematic strictly based on the taxonomy. If problematic, return the category title you find it to be belonging to as the key value. Otherwise return an empty string as the key value. Analyse all possible sub ad images present as well."""
                        },
                        {
                            "type": "text",
                            "text": f"{ocr_text}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
            },
            {
                "role": "system",
                "content":
                    """You are an expert in evaluating online display advertisements based on a taxonomy derived from ad exchange policies and user perception studies. Analyze the provided advertisement screenshot to determine if it is problematic, strictly using the provided taxonomy. For the ad images, you should also identify advertiser information. 
                    ##
                    **Advertiser information**:
                    Extract the advertiser's name from visible brand logos, text, or symbols in the ad image -- looking for unique fonts, stylized text, distinct colors, or logo designs that stand out from the main content.
                    - For Taboola and Outbrain ads, consider the advertiser as the publisher's name below each ad title. 
                    - Ads with obscure advertiser names that resemble english sentences should be categorized as *logo*.
                    - Ad screenshots with multiple advertisers should be classified as *multiple advertisers*. 
                    ## 
                    The text present (OCR output) in the ad creative is also provided as an additional input for processing and identifying any problematic behaviour.
                    ##
                    The taxonomy includes a title, definition, keywords, and notes for each category. Strictly use the definitions, keywords, and notes as primary guides. The keywords are exhaustive and represent the themes covered in each category.  
                    ##  
                    Be conservative in your classification: Only classify if the ad clearly and confidently fits a category. If in doubt, classify it as non-problematic.   
                    ##
                    Since it is a multi-label classification, multiple categories can be chosen. Return a JSON object with the following structure:  
                    1. A `categories` key containing a list of applicable categories. If non-problematic, return an empty list as the value.  
                    2. An `explanations` key providing a rationale for each identified category. Include detailed reasoning for each flagged category.
                    3. An `advertiser` key containing the advertiser name identified in the ad image. If multiple ads exist, report 'multiple advertiser'. If no advertiser identification is possible or no subtitle is present, report *None* in the value.  
                    ##  
                    For screenshots with multiple ads, return the superset of categories covering all ads.
                    ###
                    Taxonomy:
                    1. Regulations
                    Definition: Ads deemed inappropriate for younger audiences (Below 18 years) due to product or content nature. 
                    Keywords: sexually suggestive, Cannabinoids, alcohol, gambling/sweepstakes, cosmetic interventions, hazardous goods, drug use, Weight-reduction regimes or establishments, Lottery, E-Cigarettes, Prescription-only medicines, Dating sites
                    #
                    Note: 
                    - Classify only if clear evidence like explicit exposure, provocative poses, or overtly suggestive context exists; avoid bias towards modeling images. 
                    - Sexually suggestive ads should be classified into both `Regulations` and `Inappropriate or Offensive Content` if the content is derogatory or explicit.
                    ##
                    2. Inappropriate or Offensive Content
                    Definition: Ads containing language, visuals, or themes that are offensive, hateful, or disrespectful towards general public and not focused groups.
                    Keywords: Sexually explicit, offensive language, violent acts, hate speech, hookup, graphic images, racially insensitive, conspiracy theories, disrespectful religious/sacred content/profanity, trafficking, social issues, hacking
                    #
                    Note: 
                    - Merely mentioning potentially sensitive topics (e.g., violence, religion) does not automatically make the ad offensive unless it is presented in a disturbing, demeaning, or shocking manner.
                    - References to advertiser or domain names that contain edgy words or horror themes are not automatically classified as offensive.
                    ##
                    3. Deceptive Claims and Exaggerated Benefits 
                    Definition: Ads that make unverified or exaggerated claims about a product's or service's effectiveness, often with the intent to mislead consumers. Unlike Dark Patterns, these ads include specific claims in their content, rather than relying solely on clickbait tactics.
                    Keywords: 
                        Health claims - No disclosure, Reasons for conditions, miracle cure, weight loss, scientifically proven, Dietary supplements, cosmetic beauty treatments, mental health services, false vaccine information, cheap substitutes
                        Financial Claims - get rich quick, financial freedom, investment returns, guaranteed profits, False Tax Promises, Crypto Gain Misrepresentation, debt relief
                        Environmental Claims - No disclosure, greenwashing (eco-friendly, sustainable, green product, carbon-neutral, environmentally safe), eco-friendly, ethical sourcing, organic, waste reduction claims
                        Other Impossible Claims - overpromising, instant results, transform your life, best in the world/market, one-of-a-kind, never-before-seen, guaranteed satisfaction, exclusive deal, legal claims
                    #
                    Note: 
                    - No disclosures are required for finance-related ads offering incentives like 'up to a certain amount' for account openings. 
                    - Focus on identifying financial claims that appear fraudulent or excessively unrealistic, rather than scrutinizing all claims.
                    ##
                    4. Dark Patterns and Manipulative Design
                    Definition: Ads that use deceptive design techniques to manipulate user behavior, such as clicking, subscribing, or sharing information unintentionally.
                    Keywords: clickbait, social engineering, scarcity tactics, confirmshaming, countdown timers, fake buttons, sensationalism, fake testimonials, fake celebrity endorsements, urgency, last chance, your data is at risk, fear tactics, emergency, danger, don't miss out, Incomplete sentences using ...
                    #
                    Note: 
                    - Fake or multiple buttons present in the Ad image should be classified as *Dark Patterns and Manipulative Design*.
                    - Search ads with multiple clickable rows should be classified as *Dark Patterns and Manipulative Design* 
                    ##
                    5. User Experience Disruption  
                    Definition: Ads that degrade the user experience by being overly intrusive, misleading, repetitive, or lck of advertiser information. This includes ads with unnatural, or highly manipulated imagery that causes confusion or doubt about authenticity.  
                    Keywords: annoying, intrusive (revealing locations), auto-playing video, difficult-to-close pop-ups, disruptive ad formats, ad loading speed, unclear labeling of sponsored content. Ad quality (image), No Information (text/image) - for context.  
                    #
                    Note:
                    - Ads containing **map snippets** that indicate proximity to the user's location are considered intrusive, classify them as **User Experience Disruption**.
                    - Different ads with **identical images** within the ad screenshot, are classified them as **User Experience Disruption**. Multiple but different ads appearing in the same screenshot are not disruption.
                    - Ignore popup notifications originating from or local to the website itself.
                    ##
                    6. Political and Socially Sensitive Topics
                    Definition: Ads related to political topics that could polarize or manipulate public opinion, especially during sensitive times like elections.
                    Keywords: No disclosure (endorsements), election campaigns, fake endorsements, fake news, propaganda (climate change, LGBTQ rights, racial justice, religious freedom, abortion, immigration policy), defaming candidates
                    #
                    Note: 
                    - Classify specific propaganda ads if the content undermines trust or portrays a person or entity in a detrimental or questionable manner.
                    """
            },
            ],
            temperature=0,
            response_format={'type':'json_object'}
        )

        res = response.choices[0].message.content
    # print(res)
    except Exception as e:
        print('ERROR l261:', e)
        res = "{}"

    try:
        res = json.loads(res)
        if "explanations" not in res.keys():
            res["explanations"] = {}
        elif type(res["explanations"]) == list:
            res["explanations"] = {}


        # Add Dark patterns due to '...'
        if check_dp(ocr_text) and 'Dark Patterns and Manipulative Design' not in res['categories']:
            res["categories"].append('Dark Patterns and Manipulative Design')
            res["explanations"]['dp'] = "DP artificially added"
        
        # Add User Experience Disruption due to brand name not being present
        if res['advertiser'] == 'None' or any(re.search(pattern, ocr_text.lower()) for pattern in patterns):
            if 'User Experience Disruption' not in res["categories"]:
                res["categories"].append('User Experience Disruption')
                res["explanations"]['ued'] = 'UED artifically added'

        print(res)
        all_annot[image_folder] = ';'.join(res["categories"])
        explanations[image_folder] = res["explanations"]
    except:
        print('ERROR', res)
        all_annot[image_folder] = res
        explanations[image_folder] = res
    json.dump(all_annot, open('llm_annotation_dict_4o.json', 'w'))
    json.dump(explanations, open('llm_annotation_explanations_4o.json', 'w'))

df['llm_annotation'] = list(all_annot.values())
df['explanations'] = list(explanations.values())
df.to_csv('llm_annotation_4o.csv', index=False)