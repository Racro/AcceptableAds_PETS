"""
Sheep's clothing, wolfish impact - Research artifact for studying acceptable ads and their impact on user privacy
Copyright (C) 2025  Ritik Roongta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from PIL import Image

## Initialisation on terminal
# gcloud init
# gcloud auth application-default login

def detect_text(path):
    """Detects text in the file using Google Cloud Vision API with Tesseract fallback."""
    
    # Try Google Cloud Vision first
    try:
        from google.cloud import vision
        
        client = vision.ImageAnnotatorClient()

        with open(path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )
        
        texts = response.text_annotations
        
        ret = ''
        
        if texts == '':
            return ret        
        
        for text in texts:
            ret = ret + ' ' + text.description
        return ret
        
    except (ImportError, Exception) as e:
        # Fallback to Tesseract if Google Cloud Vision fails or is not available
        print(f"Google Cloud Vision failed: {e}")
        print("Falling back to Tesseract OCR...")
        
        try:
            import pytesseract
            
            # Open the image using PIL
            image = Image.open(path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)
            
            return text.strip()
            
        except ImportError:
            print("Tesseract not available. Please install pytesseract and tesseract-ocr.")
            return ""
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
            return ""