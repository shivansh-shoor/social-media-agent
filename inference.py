import os
import requests
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://shivanshd-meh.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
# Fix: use the env var name you expect (HF_TOKEN). Don't hardcode the token name as the key.
HF_TOKEN = os.getenv("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")

# Debug: print working dir and env info (remove or reduce in production)
print("cwd:", os.getcwd())
print("API_BASE_URL:", API_BASE_URL)
print("MODEL_NAME:", MODEL_NAME)
print("HF_TOKEN set:", bool(HF_TOKEN))

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

print("[START]")

for ep in range(1, 4):
    # call /reset and check result
    r = requests.post(f"{API_BASE_URL}/reset")
    if r.status_code != 200:
        print(f"[ERROR] /reset returned status {r.status_code}: {r.text}")
        break

    res = r.json()
    task = res.get("task_type", "unknown")
    question = res.get("observation", "")

    # create chat completion
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}]
    )
    # depending on OpenAI SDK version you may need to adjust this path
    answer = response.choices[0].message.content

    # send step and check result
    step_r = requests.post(f"{API_BASE_URL}/step", json={"action": "answer", "query": answer})
    if step_r.status_code != 200:
        print(f"[ERROR] /step returned status {step_r.status_code}: {step_r.text}")
        break

    step_res = step_r.json()
    print(f"[STEP] episode={ep} reward={step_res.get('reward', 0)}")
    time.sleep(0.8)

print("[END]")
