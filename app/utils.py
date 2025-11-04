import os
import requests

def get_embedding(text):
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # get from environment
    model = "sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    response = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
        headers=headers,
        json={"inputs": text}
    )
    if response.status_code != 200:
        print("❌ Hugging Face API error:", response.text)
        return None
    return response.json()
