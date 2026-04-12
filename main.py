from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json, uuid, random, os

app = FastAPI(title="Social Memory OpenEnv", version="1.0.0")

DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data.json")
with open(DATA_PATH) as f:
    DATA = json.load(f)

class Action(BaseModel):
    action: str
    query: str

class StepResult(BaseModel):
    reward: float
    done: bool = False
    observation: str
    info: Dict[str, Any]

_state: Dict[str, Any] = {
    "episode_id": None,
    "task_type": None,
    "step_count": 0,
    "total_reward": 0.0,
    "max_steps": 5,
}

TASK_BANK = {
    "fact_extraction": [
        {"q": "Who went to Goa with Shivansh?", "key_facts": ["rahul", "priya"]},
        {"q": "What company is Shivansh interning at?", "key_facts": ["ecofuelz"]}
    ],
    "thread_summary": [
        {"q": "Summarize the Goa trip planning thread", "gold_keywords": ["goa", "priya", "rahul", "beach", "hotel"]}
    ],
    "multi_hop_qa": [
        {"q": "What gift is planned for Priya's birthday and who contributes how much?", "key_facts": ["necklace", "2000", "3000"]}
    ]
}

def _build_obs(tt: str, q: dict) -> str:
    return f"TASK: {tt.upper()}\nQuestion: {q['q']}\n\nUse the provided social memory to answer."

@app.post("/reset")
async def reset():
    tt = random.choice(list(TASK_BANK.keys()))
    q = random.choice(TASK_BANK[tt])
    _state.update({
        "episode_id": str(uuid.uuid4()),
        "step_count": 0,
        "total_reward": 0.0,
        "task_type": tt,
        "current_question": q
    })
    return {
        "observation": _build_obs(tt, q),
        "episode_id": _state["episode_id"],
        "task_type": tt,
        "reward": 0.0,
        "done": False
    }

@app.post("/step")
async def step(action: Action):
    if not _state["episode_id"]:
        raise HTTPException(status_code=400, detail="Call /reset first")
    _state["step_count"] += 1
    text = action.query.lower()
    q = _state["current_question"]
    if _state["task_type"] == "fact_extraction":
        reward = 1.0 if any(k in text for k in q.get("key_facts", [])) else 0.4
    elif _state["task_type"] == "thread_summary":
        reward = 0.8 if len(text.split()) > 15 else 0.5
    else:
        reward = 1.0 if any(k in text for k in q.get("key_facts", [])) else 0.5
    _state["total_reward"] += reward
    done = _state["step_count"] >= _state["max_steps"]
    return StepResult(
        reward=round(reward, 4),
        done=done,
        observation=f"Step {_state['step_count']}: reward = {reward:.2f}",
        info={"task": _state["task_type"], "total_reward": round(_state["total_reward"], 4)}
    )

@app.get("/state")
async def get_state():
    return {"episode_id": _state.get("episode_id"), "task": _state.get("task_type")}

@app.get("/health")
async def health():
    return {"status": "ok"}
