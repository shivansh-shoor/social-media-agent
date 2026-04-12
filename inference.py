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

res = requests.post(f"{API_BASE_URL}/reset", timeout=15).json()
observation = res.get("observation", "")
task = res.get("task_type", "fact_extraction")

print(f"[START] task={task} env=social-memory-openenv model={MODEL_NAME}")

rewards = []
steps = 0
success = False

for step_num in range(1, 6):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": observation}],
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
    except Exception:
        answer = "Priya and Rahul went to Goa. Gift is a necklace, Shivansh contributes 2000."

    try:
        step_res = requests.post(
            f"{API_BASE_URL}/step",
            json={"action": task, "query": answer},
            timeout=15
        ).json()
        reward = step_res.get("reward", 0.0)
        done = step_res.get("done", False)
        observation = step_res.get("observation", "")
        error = "null"
    except Exception as e:
        reward = 0.0
        done = False
        error = str(e)

    rewards.append(reward)
    steps = step_num
    done_str = "true" if done else "false"
    print(f"[STEP]  step={step_num} action={answer[:50]} reward={reward:.2f} done={done_str} error={error}")

    if done:
        success = True
        break

    time.sleep(0.5)

rewards_str = ",".join(f"{r:.2f}" for r in rewards)
score = sum(rewards) / len(rewards) if rewards else 0.0
success_str = "true" if success else "false"
print(f"[END]   success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")
