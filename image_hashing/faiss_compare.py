import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

def return_index(image_path, index):
    image = Image.open(image_path)

    # Load the model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

    # Extract the features
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    # Normalize the features before search
    embeddings = outputs.last_hidden_state
    embeddings = embeddings.mean(dim=1)
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)

    # Read the index file and perform search of top-3 images
    d,i = index.search(vector,10)
    return d, i