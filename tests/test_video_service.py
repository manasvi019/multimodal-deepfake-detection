
import cv2
import numpy as np

from services.video_service import get_video_info
import pytest
import services.upload_service as upload_service


@pytest.fixture
def temporary_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_service, "UPLOAD_DIR", tmp_path)
    return tmp_path



def test_get_video_info_from_valid_video(temporary_upload_dir):
    video_path = temporary_upload_dir / "sample.mp4"

    # Create a small, valid video for the test.
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )

    if not writer.isOpened():
        writer.release()
        import pytest
        pytest.skip("OpenCV MP4 video writer is unavailable.")

    try:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        for _ in range(5):
            writer.write(frame)
    finally:
        writer.release()

    result = get_video_info("sample.mp4")

    assert result is not None
    assert result["fps"] > 0
    assert result["total_frames"] > 0
    assert result["width"] == 64
    assert result["height"] == 48
    assert result["duration_seconds"] > 0
