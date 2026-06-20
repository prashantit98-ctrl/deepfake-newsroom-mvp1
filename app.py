from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def home():
    return {
        "project": "Deepfake Newsroom MVP",
        "status": "online"
    }
