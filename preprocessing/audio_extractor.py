
import subprocess
from pathlib import Path


def extract_audio(video_path, output_path):
    """Extract audio from a video and save it as WAV."""

    video_path = Path(video_path) #og video path
    output_path = Path(output_path) #audio storing path

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-y", str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")

    return str(output_path)
