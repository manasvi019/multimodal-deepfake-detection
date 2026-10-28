from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentInput:
    segment_id: str
    start_sec: float
    end_sec: float

    # RGB uint8 frames: (num_frames, height, width, 3)
    video_frames: NDArray[np.uint8]

    # One original-timeline timestamp per frame.
    frame_timestamps_sec: NDArray[np.float64]

    # Mono float32 audio at 16 kHz, or None if unavailable.
    audio_samples: Optional[NDArray[np.float32]]

    audio_sample_rate: int = 16000
    video_available: bool = True
    audio_available: bool = True
    audio_start_sec: Optional[float] = None
    audio_end_sec: Optional[float] = None