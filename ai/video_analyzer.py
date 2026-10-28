from pathlib import Path

import numpy as np
import timm
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from timm.data import create_transform, resolve_data_config

import train_video_xception_cpu as training


class VideoAnalyzer:
    def __init__(self):
        project_dir = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            project_dir / "artifacts" / "video_xception_full" / "classifier.pt"
        )
        face_model_path = project_dir / "models" / "blaze_face_short_range.tflite"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
        if not face_model_path.is_file():
            raise FileNotFoundError(f"Face detector model not found: {face_model_path}")

        self.feature_model = timm.create_model(
            training.MODEL_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_model.eval()
        for parameter in self.feature_model.parameters():
            parameter.requires_grad_(False)

        self.transform = create_transform(
            **resolve_data_config({}, model=self.feature_model),
            is_training=False,
        )

        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(face_model_path)),
            min_detection_confidence=0.3,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.classifier = training.SmallClassifier(checkpoint["input_dim"])
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()

    @torch.inference_mode()
    def predict_frames(self, frames_rgb: np.ndarray) -> float | None:
        features = training.extract_segment_feature(
            frames_rgb,
            self.feature_model,
            self.transform,
            self.detector,
            sample_count=3,
        )
        if features is None:
            return None

        feature_tensor = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        ).reshape(1, -1)

        return float(torch.sigmoid(self.classifier(feature_tensor)).item())

    def close(self) -> None:
        self.detector.close()