
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
