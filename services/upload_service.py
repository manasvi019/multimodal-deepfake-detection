
import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

# Provisional defaults; confirm with the team before finalizing.
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MiB
CHUNK_SIZE = 1024 * 1024  # 1 MiB

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"


class UploadValidationError(Exception):
    """Raised when an uploaded video fails validation."""


def has_supported_video_signature(header: bytes, extension: str) -> bool:
    """Check common container signatures; this does not fully decode a video."""
    if extension == ".mp4":
        # ISO Base Media File Format commonly uses an ftyp box near the start.
        return len(header) >= 8 and header[4:8] == b"ftyp"

    if extension == ".mov":
        # MOV commonly uses the same ISO Base Media File Format container.
        return len(header) >= 8 and header[4:8] == b"ftyp"

    if extension == ".avi":
        # AVI files start with RIFF and identify the AVI form at byte 8.
        return (
            len(header) >= 12
            and header[:4] == b"RIFF"
            and header[8:12] == b"AVI "
        )

    return False



async def save_video(upload: UploadFile) -> dict:
    """Validate and store an uploaded video using a server-generated name."""
    original_name = Path(upload.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name or extension not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            "Unsupported file extension. Allowed formats: MP4, MOV, AVI."
        )

    # Preliminary MIME check only; MIME type is client-supplied.
    allowed_mime_types = {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "application/octet-stream",
    }
    if upload.content_type not in allowed_mime_types:
        raise UploadValidationError("Unsupported content type.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    total_size = 0

    try:
        with destination.open("xb") as output:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break

                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise UploadValidationError(
                        "File exceeds the provisional 100 MiB upload limit."
                    )

                output.write(chunk)

        if total_size == 0:
            raise UploadValidationError("Uploaded file is empty.")
        with destination.open("rb") as saved_file:
            header = saved_file.read(32)

        if not has_supported_video_signature(header, extension):
            raise UploadValidationError(
                "File content does not match a supported video format."
                )
        return {
            "filename": destination.name,
            "size_bytes": total_size,
            "message": "Video uploaded successfully.",
        }

    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
def list_uploaded_videos() -> list[dict]:
    """Return metadata for videos stored in the uploads directory."""
    if not UPLOAD_DIR.exists():
        return []

    videos = []

    for path in UPLOAD_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            stat = path.stat()
            videos.append({
                "filename": path.name,
                "size_bytes": stat.st_size,
            })

    return sorted(videos, key=lambda video: video["filename"])

def get_video_details(filename: str) -> dict | None:
    """Return metadata for one uploaded video."""
    if not filename or Path(filename).name != filename:
        return None

    path = UPLOAD_DIR / filename

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None

    if not path.is_file():
        return None

    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
    }