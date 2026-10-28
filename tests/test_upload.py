
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
                b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32,
                "video/mp4",
            )
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Video uploaded successfully."
    assert data["size_bytes"] == len(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
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

def test_reject_mp4_with_invalid_content(temporary_upload_dir):
    response = client.post(
        "/upload",
        files={
            "file": (
                "fake.mp4",
                b"this is not a video",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 400
    assert "content" in response.json()["detail"].lower()
    assert list(temporary_upload_dir.iterdir()) == []

def test_list_uploaded_videos(temporary_upload_dir):
    video_file = temporary_upload_dir / "sample.mp4"
    video_file.write_bytes(b"sample video content")

    response = client.get("/videos")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["videos"] == [
        {
            "filename": "sample.mp4",
            "size_bytes": len(b"sample video content"),
        }
    ]


def test_list_uploaded_videos_empty(temporary_upload_dir):
    response = client.get("/videos")

    assert response.status_code == 200
    assert response.json() == {
        "count": 0,
        "videos": [],
    }


def test_list_uploaded_videos_missing_directory(
    temporary_upload_dir,
    monkeypatch,
):
    # Point the service to a directory that does not exist.
    missing_dir = temporary_upload_dir / "missing"
    monkeypatch.setattr(upload_service, "UPLOAD_DIR", missing_dir)

    response = client.get("/videos")

    assert response.status_code == 200
    assert response.json() == {
        "count": 0,
        "videos": [],
    }

def test_get_video_details(temporary_upload_dir):
    video_content = b"sample video content"
    video_file = temporary_upload_dir / "sample.mp4"
    video_file.write_bytes(video_content)

    response = client.get("/videos/sample.mp4")

    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.mp4",
        "size_bytes": len(video_content),
    }


def test_get_video_details_not_found(temporary_upload_dir):
    response = client.get("/videos/unknown.mp4")

    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found."


def test_get_video_details_reject_invalid_path(temporary_upload_dir):
    response = client.get("/videos/..%2Fsecret.mp4")

    assert response.status_code == 404
    assert response.json()["detail"] in {
        "Video not found.",
        "Not Found",
    }

