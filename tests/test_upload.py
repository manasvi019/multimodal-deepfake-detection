
import pytest
from fastapi.testclient import TestClient

import services.upload_service as upload_service
from main import app

client = TestClient(app)


@pytest.fixture
def temporary_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_service, "UPLOAD_DIR", tmp_path)
    return tmp_path


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_valid_mp4(temporary_upload_dir):
    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.mp4",
                b"sample video content",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Video uploaded successfully."
    assert data["size_bytes"] == len(b"sample video content")
    assert data["filename"].endswith(".mp4")
    assert (temporary_upload_dir / data["filename"]).exists()


def test_reject_unsupported_extension(temporary_upload_dir):
    response = client.post(
        "/upload",
        files={
            "file": (
                "picture.png",
                b"not a video",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]
    assert list(temporary_upload_dir.iterdir()) == []


def test_reject_empty_file(temporary_upload_dir):
    response = client.post(
        "/upload",
        files={
            "file": ("empty.mp4", b"", "video/mp4")
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."
    assert list(temporary_upload_dir.iterdir()) == []


def test_reject_oversized_file(temporary_upload_dir, monkeypatch):
    monkeypatch.setattr(upload_service, "MAX_FILE_SIZE", 10)

    response = client.post(
        "/upload",
        files={
            "file": (
                "large.mp4",
                b"x" * 11,
                "video/mp4",
            )
        },
    )

    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"]
    assert list(temporary_upload_dir.iterdir()) == []
