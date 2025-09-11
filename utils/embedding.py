import requests
import numpy as np 
import os 
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("EURI_API_KEY")

def generate_embeddings(text):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    embedding = np.array(data['data'][0]['embedding'])
    
    return embedding