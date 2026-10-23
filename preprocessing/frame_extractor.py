
import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, every_n_frames=30):
    """Extract one frame at a fixed interval from a video."""

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_index % every_n_frames == 0:
            output_path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_index += 1

    video.release()
    return saved_count
