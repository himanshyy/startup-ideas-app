import requests
import os

# Model endpoint (you can change model name if needed)
HF_MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

def get_embedding(text):
    """
    Uses Hugging Face API to generate embeddings for a given text.
    Lightweight, safe for Render free tier.
    """
    response = requests.post(HF_MODEL_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        print("❌ Error:", response.text)
        return None
    
    return response.json()[0]
