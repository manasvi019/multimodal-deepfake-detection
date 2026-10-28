from dataclasses import dataclass

import numpy as np
from preprocessing.media_types import SegmentInput

from .audio_analyzer import AudioAnalyzer
from .video_analyzer import VideoAnalyzer


@dataclass
class SegmentAnalysis:
    segment_id: str
    start_sec: float
    end_sec: float
    audio_fake_score: float | None
    video_fake_score: float | None
    sync_mismatch_score: float | None = None


@dataclass
class AnalysisResult:
    status: str
    prediction: str
    confidence: float | None
    segments: list[SegmentAnalysis]
    errors: list[dict[str, str]]


def analyze_segments(segments: list[SegmentInput]) -> AnalysisResult:
    errors: list[dict[str, str]] = []
    results: list[SegmentAnalysis] = []

    if not segments:
        return AnalysisResult(
            status="failed",
            prediction="inconclusive",
            confidence=None,
            segments=[],
            errors=[{"analyzer": "pipeline", "message": "No segments were provided."}],
        )

    try:
        video_analyzer = VideoAnalyzer()
    except Exception as exc:
        video_analyzer = None
        errors.append({"analyzer": "video", "message": str(exc)})

    try:
        audio_analyzer = AudioAnalyzer()
    except Exception as exc:
        audio_analyzer = None
        errors.append({"analyzer": "audio", "message": str(exc)})

    errors.append({
        "analyzer": "sync",
        "message": "Synchronization analysis is not implemented yet.",
    })

    try:
        for segment in segments:
            video_score = None
            audio_score = None

            if video_analyzer is not None and getattr(segment, "video_available", True):
                try:
                    frames = getattr(segment, "video_frames", None)
                    if frames is not None:
                        video_score = video_analyzer.predict_frames(np.asarray(frames))
                except Exception as exc:
                    errors.append({
                        "analyzer": "video",
                        "segment_id": str(segment.segment_id),
                        "message": str(exc),
                    })

            if audio_analyzer is not None and getattr(segment, "audio_available", True):
                try:
                    samples = getattr(segment, "audio_samples", None)
                    if samples is not None:
                        audio_score = audio_analyzer.predict_samples(
                            samples,
                            sample_rate=getattr(segment, "audio_sample_rate", 16000),
                        )
                except Exception as exc:
                    errors.append({
                        "analyzer": "audio",
                        "segment_id": str(segment.segment_id),
                        "message": str(exc),
                    })

            results.append(
                SegmentAnalysis(
                    segment_id=str(segment.segment_id),
                    start_sec=float(segment.start_sec),
                    end_sec=float(segment.end_sec),
                    audio_fake_score=audio_score,
                    video_fake_score=video_score,
                    sync_mismatch_score=None,
                )
            )
    finally:
        if video_analyzer is not None:
            video_analyzer.close()

    return AnalysisResult(
        status="partial",
        prediction="inconclusive",
        confidence=None,
        segments=results,
        errors=errors,
    )