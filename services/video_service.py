
import cv2
from pathlib import Path

from preprocessing.video_reader import read_video_info
from services import upload_service

def get_video_info(filename: str) -> dict | None:
    """Return technical metadata for an uploaded video."""
    if not filename or Path(filename).name != filename:
        return None

    video_path = upload_service.UPLOAD_DIR / filename

    if video_path.suffix.lower() not in upload_service.ALLOWED_EXTENSIONS:
        return None

    if not video_path.is_file():
        return None

    try:
        return read_video_info(video_path)
    except (ValueError, cv2.error):
        return None
