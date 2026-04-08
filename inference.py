import os
import requests
import time
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_BASE_URL = os.getenv("API_BASE_URL", "https://shivanshd-meh.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
# Use the expected environment variable name HF_TOKEN
HF_TOKEN = os.getenv("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")

logging.info("cwd: %s", os.getcwd())
logging.info("API_BASE_URL: %s", API_BASE_URL)
logging.info("MODEL_NAME: %s", MODEL_NAME)
logging.info("HF_TOKEN set: %s", bool(HF_TOKEN))

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

logging.info("[START]")

for ep in range(1, 4):
    # call /reset and check result
    try:
        r = requests.post(f"{API_BASE_URL}/reset", timeout=10)
        r.raise_for_status()
    except Exception as e:
        logging.error("[ERROR] /reset failed: %s", e)
        break

    try:
        res = r.json()
    except ValueError:
        logging.error("[ERROR] /reset returned non-JSON response: %s", r.text)
        break

    task = res.get("task_type", "unknown")
    question = res.get("observation", "")

    # create chat completion
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": question}]
        )
    except Exception as e:
        logging.error("[ERROR] OpenAI request failed: %s", e)
        break

    # handle different SDK response shapes
    answer = None
    try:
        # newer SDKs: response.choices[0].message.content
        answer = response.choices[0].message.content
    except Exception:
        try:
            # some SDKs: response.choices[0].text
            answer = response.choices[0].text
        except Exception:
            # fallback to stringifying the response
            answer = str(response)

    # send step and check result
    try:
        step_r = requests.post(f"{API_BASE_URL}/step", json={"action": "answer", "query": answer}, timeout=10)
        step_r.raise_for_status()
    except Exception as e:
        logging.error("[ERROR] /step failed: %s", e)
        break

    try:
        step_res = step_r.json()
    except ValueError:
        logging.error("[ERROR] /step returned non-JSON response: %s", step_r.text)
        break

    logging.info("[STEP] episode=%d reward=%s", ep, step_res.get('reward', 0))
    time.sleep(0.8)

logging.info("[END]")
