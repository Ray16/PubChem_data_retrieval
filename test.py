import os
import json
from openai import OpenAI
from tqdm import tqdm
import pandas as pd 

ARGO_PORT = 8081

client = OpenAI(
    api_key="YOUR_API_KEY", # Use your actual API key here
    base_url=f"http://127.0.0.1:{ARGO_PORT}/v1"
)

#os.makedirs('answers', exist_ok=True)
#os.makedirs('output', exist_ok=True) 

prompt = open('Prompt.txt').read()

response = client.chat.completions.create(
    model="argo:gpt-4large",
    messages=[
        {"role": "user", "content": f"{prompt}"}
    ]
)

llm_answer = response.choices[0].message.content
print(llm_answer)