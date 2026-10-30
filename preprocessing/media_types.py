
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentInput:
    """Time-aligned video and audio data for one media segment."""

    segment_id: str
    start_sec: float
    end_sec: float

    # RGB frames: (num_frames, height, width, 3), uint8
    video_frames: NDArray[np.uint8]

    # One timestamp per frame, relative to the original video.
    frame_timestamps_sec: NDArray[np.float64]

    # Mono audio: (num_samples,), float32, 16 kHz.
    # None when audio is unavailable.
    audio_samples: Optional[NDArray[np.float32]]

    audio_sample_rate: int = 16000

    video_available: bool = True
    audio_available: bool = True

    audio_start_sec: Optional[float] = None
    audio_end_sec: Optional[float] = None

    def __post_init__(self):
        if not self.segment_id:
            raise ValueError("segment_id cannot be empty")

        if not (
            np.isfinite(self.start_sec)
            and np.isfinite(self.end_sec)
            and 0 <= self.start_sec < self.end_sec
        ):
            raise ValueError("Invalid segment boundaries")

        if self.audio_sample_rate != 16000:
            raise ValueError("Audio sample rate must be 16000 Hz")

        # Validate video data.
        if not isinstance(self.video_frames, np.ndarray):
            raise ValueError("video_frames must be a NumPy array")

        if self.video_frames.ndim != 4:
            raise ValueError(
                "video_frames must have shape "
                "(num_frames, height, width, 3)"
            )

        if self.video_frames.shape[-1] != 3:
            raise ValueError("Video frames must use RGB format")

        if self.video_frames.dtype != np.uint8:
            raise ValueError("Video frames must have uint8 dtype")

        if not isinstance(self.frame_timestamps_sec, np.ndarray):
            raise ValueError("Frame timestamps must be a NumPy array")

        if self.frame_timestamps_sec.ndim != 1:
            raise ValueError("Frame timestamps must be one-dimensional")

        if len(self.frame_timestamps_sec) != len(self.video_frames):
            raise ValueError("Frame count and timestamp count must match")

        if not np.all(np.isfinite(self.frame_timestamps_sec)):
            raise ValueError("Frame timestamps must be finite")

        if np.any(np.diff(self.frame_timestamps_sec) < 0):
            raise ValueError("Frame timestamps must be ordered")

        if len(self.frame_timestamps_sec):
            if np.any(self.frame_timestamps_sec < self.start_sec):
                raise ValueError("Frame timestamp before segment start")

            if np.any(self.frame_timestamps_sec > self.end_sec):
                raise ValueError("Frame timestamp after segment end")

        if self.video_available and len(self.video_frames) == 0:
            raise ValueError("Available video must contain frames")

        if not self.video_available and len(self.video_frames) != 0:
            raise ValueError("Unavailable video must contain no frames")

        # Validate audio data.
        if self.audio_available:
            if self.audio_samples is None:
                raise ValueError("Available audio requires audio_samples")

            if self.audio_samples.ndim != 1:
                raise ValueError("Audio samples must be mono")

            if self.audio_samples.dtype != np.float32:
                raise ValueError("Audio samples must have float32 dtype")

            if not np.all(np.isfinite(self.audio_samples)):
                raise ValueError("Audio samples must be finite")

            if len(self.audio_samples) == 0:
                raise ValueError("Available audio cannot be empty")

            if self.audio_start_sec is None or self.audio_end_sec is None:
                raise ValueError("Audio interval boundaries are required")

            if not (
                np.isfinite(self.audio_start_sec)
                and np.isfinite(self.audio_end_sec)
                and self.start_sec <= self.audio_start_sec
                < self.audio_end_sec <= self.end_sec
            ):
                raise ValueError("Invalid audio interval")

        else:
            if self.audio_samples is not None:
                raise ValueError(
                    "Unavailable audio must have audio_samples=None"
                )

            if (
                self.audio_start_sec is not None
                or self.audio_end_sec is not None
            ):
                raise ValueError(
                    "Unavailable audio must not have audio boundaries"
                )
