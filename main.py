
from fastapi import FastAPI

app = FastAPI(
    title="Multi-Modal Deepfake Detection API",
    description="Backend API for the deepfake detection project",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
