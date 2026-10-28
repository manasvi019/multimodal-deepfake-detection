
import cv2


def read_video_info(video_path):
    """Read basic information from a video file."""
    video = cv2.VideoCapture(str(video_path))

    try:
        if not video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = total_frames / fps if fps > 0 else 0

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }
    finally:
        video.release()
