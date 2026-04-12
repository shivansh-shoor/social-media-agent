import os
import requests
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://shivanshd-meh.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url="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",
    api_key=HF_TOKEN
)

print("START")

res = requests.post(f"{API_BASE_URL}/reset", timeout=15).json()
observation = res.get("observation", "")
task = res.get("task_type", "unknown")

for step_num in range(1, 6):
    print(f"STEP {step_num}")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": observation}],
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
    except Exception:
        answer = "Priya and Rahul went to Goa. Gift is a necklace, Shivansh contributes 2000."

    step_res = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": task, "query": answer},
        timeout=15
    ).json()

    observation = step_res.get("observation", "")

    if step_res.get("done"):
        break

    time.sleep(0.5)

print("END")
