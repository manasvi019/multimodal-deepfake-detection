
import subprocess

import numpy as np
import pytest

from preprocessing.media_preprocessor import (
    MediaPreprocessor,
    MediaPreprocessorConfig,
)


@pytest.fixture
def media_preprocessor():
    return MediaPreprocessor()


@pytest.fixture
def video_with_audio(tmp_path):
    video_path = tmp_path / "video_with_audio.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=size=320x240:rate=10:duration=9",
            "-f", "lavfi",
            "-i", "sine=frequency=1000:sample_rate=44100:duration=9",
            "-shortest",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    return video_path


@pytest.fixture
def video_without_audio(tmp_path):
    video_path = tmp_path / "video_without_audio.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=size=320x240:rate=10:duration=6",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    return video_path


def test_segment_input_configuration(media_preprocessor):
    config = media_preprocessor.config

    assert config.segment_duration_sec == 5.0
    assert config.overlap_sec == 1.0
    assert config.audio_sample_rate == 16000


def test_video_with_audio_is_segmented(
    media_preprocessor, video_with_audio
):
    segments = media_preprocessor.preprocess(video_with_audio)

    assert len(segments) == 3

    assert [(s.start_sec, s.end_sec) for s in segments] == [
        (0.0, 5.0),
        (4.0, 9.0),
        (8.0, 9.0),
    ]

    assert [len(s.video_frames) for s in segments] == [
        50, 50, 10
    ]


def test_audio_extraction_properties(
    media_preprocessor, video_with_audio
):
    segments = media_preprocessor.preprocess(video_with_audio)

    for segment in segments:
        assert segment.audio_available is True
        assert segment.audio_samples is not None
        assert segment.audio_samples.dtype == np.float32
        assert segment.audio_samples.ndim == 1
        assert segment.audio_sample_rate == 16000


def test_video_frame_properties(
    media_preprocessor, video_with_audio
):
    segments = media_preprocessor.preprocess(video_with_audio)

    for segment in segments:
        assert segment.video_available is True
        assert segment.video_frames.dtype == np.uint8
        assert segment.video_frames.ndim == 4
        assert segment.video_frames.shape[-1] == 3
        assert segment.frame_timestamps_sec.dtype == np.float64
        assert len(segment.frame_timestamps_sec) == len(
            segment.video_frames
        )


def test_video_without_audio(
    media_preprocessor, video_without_audio
):
    segments = media_preprocessor.preprocess(video_without_audio)

    assert len(segments) == 2

    for segment in segments:
        assert segment.video_available is True
        assert segment.audio_available is False
        assert segment.audio_samples is None


def test_invalid_segment_configuration():
    with pytest.raises(ValueError):
        MediaPreprocessorConfig(
            segment_duration_sec=0
        )

    with pytest.raises(ValueError):
        MediaPreprocessorConfig(
            segment_duration_sec=5,
            overlap_sec=5,
        )

    with pytest.raises(ValueError):
        MediaPreprocessorConfig(
            audio_sample_rate=44100
        )


def test_missing_video_file(media_preprocessor, tmp_path):
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        media_preprocessor.preprocess(missing_path)
