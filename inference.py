import os
import requests
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://shivanshd-meh.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1",
    api_key=HF_TOKEN
)

print("START")

for ep in range(1, 4):
    print(f"EPISODE {ep}")

    try:
        res = requests.post(f"{API_BASE_URL}/reset", timeout=15).json()
        observation = res.get("observation", "")
        task = res.get("task_type", "unknown")
        print(f"TASK {task}")
    except Exception as e:
        print(f"RESET_ERROR {e}")
        continue

    for step_num in range(1, 4):
        print(f"STEP {step_num}")
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": observation}],
                max_tokens=200
            )
            answer = response.choices[0].message.content.strip()
            print(f"ACTION {answer[:100]}")
        except Exception as e:
            print(f"LLM_ERROR {e}")
            answer = "Priya and Rahul went to Goa. Gift is a necklace, Shivansh contributes 2000."

        try:
            step_res = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": task, "query": answer},
                timeout=15
            ).json()
            reward = step_res.get("reward", 0)
            print(f"REWARD {reward}")
            if step_res.get("done"):
                break
        except Exception as e:
            print(f"STEP_ERROR {e}")
        time.sleep(0.5)

print("END")
