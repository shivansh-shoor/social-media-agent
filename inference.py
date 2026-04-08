import os
import requests
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://shivanshd-meh.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "your-active-model")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

print("START")

for ep in range(1, 4):
    res = requests.post(f"{API_BASE_URL}/reset").json()
    task = res.get("task_type", "unknown")
    question = res.get("observation", "")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}]
    )
    answer = response.choices[0].message.content

    step_res = requests.post(f"{API_BASE_URL}/step", json={"action": "answer", "query": answer}).json()

    print(f"STEP episode={ep} reward={step_res.get('reward', 0)}")
    time.sleep(0.8)

print("END")
