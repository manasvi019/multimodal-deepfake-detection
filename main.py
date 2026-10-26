
from fastapi import FastAPI, File, HTTPException, UploadFile

from services.upload_service import (
    UploadValidationError,
    save_video,
    list_uploaded_videos,
)

app = FastAPI(
    title="Multi-Modal Deepfake Detection API",
    description="Backend API for the deepfake detection project",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload", status_code=201)
async def upload_video(file: UploadFile = File(...)):
    try:
        result = await save_video(file)
        return result

    except UploadValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Could not store the uploaded video.",
        ) from None
@app.get("/videos")
def get_uploaded_videos():
    videos = list_uploaded_videos()

    return {
        "count": len(videos),
        "videos": videos,
    }