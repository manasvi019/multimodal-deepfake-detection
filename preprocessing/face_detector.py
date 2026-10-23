
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("models/blaze_face_short_range.tflite")


def detect_faces(image_path):
    """Detect faces in an image using MediaPipe BlazeFace."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Face detector model not found: {MODEL_PATH}")

    image = mp.Image.create_from_file(str(image_path))

    options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    min_detection_confidence=0.3
    )

    with vision.FaceDetector.create_from_options(options) as detector:
        result = detector.detect(image)

    faces = []

    for detection in result.detections:
        box = detection.bounding_box

        faces.append({
            "x": box.origin_x,
            "y": box.origin_y,
            "width": box.width,
            "height": box.height,
            "score": detection.categories[0].score,
        })

    return faces
