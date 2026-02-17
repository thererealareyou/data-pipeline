import json
import os
from dotenv import load_dotenv
import requests

import config

load_dotenv()

def ask_model(context: str, prompt: str) -> str:
    url = os.getenv("URL")
    payload = {
        "model": os.getenv("MODEL"),
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "system",
                "content": f"{context}",
            },
        ],
        "temperature": config.TEMPERATURE,
        "cache_prompt": False,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, data=json.dumps(payload), headers=headers, timeout=240
        )
    except requests.exceptions.RequestException as e:
        print("Timeout")
        return "Timeout"
    if response.status_code == 200:
        print(response.json()["choices"][0]["message"]["content"])
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}")
        return "Error"
