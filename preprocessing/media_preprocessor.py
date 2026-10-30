
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preprocessing.media_types import SegmentInput


@dataclass
class MediaPreprocessorConfig:
    segment_duration_sec: float = 5.0
    overlap_sec: float = 1.0
    audio_sample_rate: int = 16000
    frame_sampling_fps: float | None = None
    ffmpeg_timeout_sec: int = 120

    def __post_init__(self):
        if self.segment_duration_sec <= 0:
            raise ValueError("Segment duration must be positive")

        if not 0 <= self.overlap_sec < self.segment_duration_sec:
            raise ValueError(
                "Overlap must be non-negative and smaller than duration"
            )

        if self.audio_sample_rate != 16000:
            raise ValueError("Audio sample rate must be 16000 Hz")

        if self.frame_sampling_fps is not None:
            if self.frame_sampling_fps <= 0:
                raise ValueError("Frame sampling FPS must be positive")


class MediaPreprocessor:
    """Split a video into time-aligned audio/video segments."""

    def __init__(self, config: MediaPreprocessorConfig | None = None):
        self.config = config or MediaPreprocessorConfig()

        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg was not found in PATH")

        if not shutil.which("ffprobe"):
            raise RuntimeError("FFprobe was not found in PATH")

    def _has_audio(self, video_path: Path) -> bool:
        """Check whether the media file contains an audio stream."""

        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(video_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.config.ffmpeg_timeout_sec,
            check=True,
        )

        data = json.loads(result.stdout)
        return bool(data.get("streams"))

    def _extract_audio(
        self,
        video_path: Path,
        start_sec: float,
        duration_sec: float,
    ) -> np.ndarray | None:
        """Extract mono 16 kHz float32 audio for one segment."""

        command = [
            "ffmpeg",
            "-v", "error",
            "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(duration_sec),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(self.config.audio_sample_rate),
            "-f", "f32le",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=self.config.ffmpeg_timeout_sec,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Audio extraction failed: "
                f"{exc.stderr.decode(errors='replace')}"
            ) from exc

        if not result.stdout:
            return None

        samples = np.frombuffer(
            result.stdout,
            dtype=np.float32,
        ).copy()

        if samples.size == 0:
            return None

        return samples

    def _extract_video_frames(
        self,
        video_path: Path,
        duration_sec: float,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Extract RGB frames and their timestamps."""

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        timestamps = []

        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS)

            if source_fps <= 0:
                raise ValueError("Could not determine video FPS")

            frame_index = 0

            while True:
                success, frame_bgr = capture.read()

                if not success:
                    break

                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Fall back to frame index / FPS if timestamp is invalid.
                if not np.isfinite(timestamp) or timestamp < 0:
                    timestamp = frame_index / source_fps

                if timestamp >= duration_sec:
                    break

                should_keep = True

                target_fps = self.config.frame_sampling_fps

                if target_fps is not None and target_fps < source_fps:
                    interval = 1.0 / target_fps
                    should_keep = (
                        not timestamps
                        or timestamp - timestamps[-1] >= interval * 0.95
                    )

                if should_keep:
                    frame_rgb = cv2.cvtColor(
                        frame_bgr,
                        cv2.COLOR_BGR2RGB,
                    )

                    frames.append(frame_rgb)
                    timestamps.append(float(timestamp))

                frame_index += 1

        finally:
            capture.release()

        if not frames:
            raise ValueError("No video frames could be extracted")

        return frames, timestamps

    def preprocess(self, video_path: str | Path) -> list[SegmentInput]:
        """Generate overlapping, time-aligned media segments."""

        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if video_path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
            raise ValueError("Unsupported video format")

        info = cv2.VideoCapture(str(video_path))

        try:
            if not info.isOpened():
                raise ValueError("Could not open video")

            fps = info.get(cv2.CAP_PROP_FPS)
            frame_count = info.get(cv2.CAP_PROP_FRAME_COUNT)

            if fps <= 0 or frame_count <= 0:
                raise ValueError("Invalid video metadata")

            duration_sec = frame_count / fps

        finally:
            info.release()

        has_audio = self._has_audio(video_path)

        frames, timestamps = self._extract_video_frames(
            video_path,
            duration_sec,
        )

        frame_times = np.asarray(timestamps, dtype=np.float64)

        step_sec = (
            self.config.segment_duration_sec
            - self.config.overlap_sec
        )

        segments = []
        segment_index = 0
        start_sec = 0.0

        while start_sec < duration_sec:
            end_sec = min(
                start_sec + self.config.segment_duration_sec,
                duration_sec,
            )

            # Use start-inclusive, end-exclusive frame intervals.
            selected_indices = np.where(
                (frame_times >= start_sec)
                & (frame_times < end_sec)
            )[0]

            if len(selected_indices) == 0:
                segment_frames = np.empty(
                    (0, 0, 0, 3),
                    dtype=np.uint8,
                )
                segment_timestamps = np.empty(
                    (0,),
                    dtype=np.float64,
                )
                video_available = False
            else:
                segment_frames = np.stack(
                    [frames[i] for i in selected_indices]
                ).astype(np.uint8, copy=False)

                segment_timestamps = frame_times[selected_indices]
                video_available = True

            audio_samples = None
            audio_start = None
            audio_end = None
            audio_available = False

            if has_audio:
                audio_samples = self._extract_audio(
                    video_path,
                    start_sec,
                    end_sec - start_sec,
                )

                if audio_samples is not None:
                    audio_start = start_sec
                    audio_end = min(
                        start_sec
                        + len(audio_samples)
                        / self.config.audio_sample_rate,
                        end_sec,
                    )

                    # Discard audio if its actual interval is invalid.
                    if audio_end > audio_start:
                        audio_available = True
                    else:
                        audio_samples = None
                        audio_start = None
                        audio_end = None

            segment = SegmentInput(
                segment_id=f"segment_{segment_index:03d}",
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                video_frames=segment_frames,
                frame_timestamps_sec=segment_timestamps,
                audio_samples=audio_samples,
                audio_sample_rate=self.config.audio_sample_rate,
                video_available=video_available,
                audio_available=audio_available,
                audio_start_sec=audio_start,
                audio_end_sec=audio_end,
            )

            segments.append(segment)

            segment_index += 1
            start_sec += step_sec

        return segments
