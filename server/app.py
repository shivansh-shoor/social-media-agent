from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/reset")
async def reset():
    return {"status": "ok", "observation": {}}

@app.post("/step")
async def step():
    return {"reward": 0.5, "done": False, "observation": {}}

@app.get("/health")
async def health():
    return {"status": "healthy"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
