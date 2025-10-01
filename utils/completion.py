import os 
import requests
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("EURI_API_KEY")

def generate_completion(prompt,model="gpt-4.1-nano", temperature=0.3):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model":model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()['choices'][0]['message']['content']
    return data