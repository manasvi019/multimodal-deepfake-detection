"""CPU-only video training/evaluation with frozen timm Xception features.

This is intentionally standalone: it does not import production inference code
and never reads or writes models/ffpp_c40.pth.  Run it from the project root
after checking the dataset paths and pretrained-weight availability.
"""

from __future__ import annotations
from PIL import Image
import argparse
from html import parser
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import timm
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose
from timm.data import create_transform, resolve_data_config


REAL = 0
FAKE = 1
MODEL_NAME = "legacy_xception.tf_in1k"
DEFAULT_OUTPUT = Path("artifacts/video_xception")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoRecord:
    path: str
    label: int
    group: str


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def identity_tokens(path: Path) -> list[str]:
    """Return identity tokens from a video stem.

    Numeric tokens support FF++ names such as 183_253.mp4.  Non-numeric names
    remain usable by treating the whole stem as one identity token.
    """
    tokens = re.findall(r"\d+", path.stem)
    return tokens or [path.stem]


def discover_records(real_dir: Path, fake_dir: Path) -> list[VideoRecord]:
    real_paths = sorted(
        p for p in real_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    fake_paths = sorted(
        p for p in fake_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not real_paths or not fake_paths:
        raise FileNotFoundError(
            f"Expected videos under {real_dir} and {fake_dir}; "
            f"found {len(real_paths)} real and {len(fake_paths)} fake videos."
        )

    uf = UnionFind()
    for path in real_paths:
        for token in identity_tokens(path):
            uf.add(token)
    for path in fake_paths:
        tokens = identity_tokens(path)
        for token in tokens:
            uf.add(token)
        for token in tokens[1:]:
            uf.union(tokens[0], token)

    def group_for(path: Path) -> str:
        roots = sorted({uf.find(token) for token in identity_tokens(path)})
        return "identity_group:" + "+".join(roots)

    return [
        *[VideoRecord(str(p), REAL, group_for(p)) for p in real_paths],
        *[VideoRecord(str(p), FAKE, group_for(p)) for p in fake_paths],
    ]


def largest_face_crop(frame_bgr: np.ndarray, detector: object) -> np.ndarray | None:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.detections:
        return None

    height, width = frame_bgr.shape[:2]
    detection = max(
        result.detections,
        key=lambda item: item.bounding_box.width * item.bounding_box.height,
    )
    box = detection.bounding_box

    x1 = max(0, box.origin_x)
    y1 = max(0, box.origin_y)
    x2 = min(width, box.origin_x + box.width)
    y2 = min(height, box.origin_y + box.height)

    return frame_bgr[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def sample_video_faces(path: Path, detector: object, sample_count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(frame_count - 1, 0), num=sample_count, dtype=int)
    faces: list[np.ndarray] = []
    for index in np.unique(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            face = largest_face_crop(frame, detector)
            if face is not None:
                faces.append(face)
    capture.release()
    return faces


def extract_video_feature(
    path: Path,
    model: nn.Module,
    transform: Compose,
    detector: object,
    sample_count: int,
) -> np.ndarray | None:
    faces = sample_video_faces(path, detector, sample_count)
    if not faces:
        return None
    batch = torch.stack(
        [transform(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))) for face in faces]
    )
    with torch.inference_mode():
        output = model(batch)
    if output.ndim > 2:
        output = output.flatten(1)
    return output.mean(dim=0).cpu().numpy().astype(np.float32)

def largest_face_crop_rgb(frame_rgb: np.ndarray, detector: object) -> np.ndarray | None:
    frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.detections:
        return None

    height, width = frame_rgb.shape[:2]
    detection = max(
        result.detections,
        key=lambda item: item.bounding_box.width * item.bounding_box.height,
    )
    box = detection.bounding_box

    x1 = max(0, box.origin_x)
    y1 = max(0, box.origin_y)
    x2 = min(width, box.origin_x + box.width)
    y2 = min(height, box.origin_y + box.height)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame_rgb[y1:y2, x1:x2]


def extract_segment_feature(
    frames_rgb: np.ndarray,
    model: nn.Module,
    transform: Compose,
    detector: object,
    sample_count: int = 3,
) -> np.ndarray | None:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError("Expected RGB frames with shape (frames, height, width, 3).")
    if frames_rgb.shape[0] == 0 or sample_count <= 0:
        return None

    indices = np.linspace(
        0,
        frames_rgb.shape[0] - 1,
        num=min(sample_count, frames_rgb.shape[0]),
        dtype=int,
    )

    faces = []
    for index in np.unique(indices):
        face = largest_face_crop_rgb(frames_rgb[index], detector)
        if face is not None:
            faces.append(face)

    if not faces:
        return None

    batch = torch.stack(
        [transform(Image.fromarray(face)) for face in faces]
    )
    with torch.inference_mode():
        output = model(batch)

    if output.ndim > 2:
        output = output.flatten(1)

    return output.mean(dim=0).cpu().numpy().astype(np.float32)


class SmallClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)


def train_classifier(
    features: np.ndarray, labels: np.ndarray, epochs: int, batch_size: int, seed: int
) -> SmallClassifier:
    torch.manual_seed(seed)
    model = SmallClassifier(features.shape[1])
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
    return model.eval()


def evaluate(model: SmallClassifier, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    if len(np.unique(labels)) != 2:
        raise ValueError("This split needs both REAL and FAKE videos.")

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(features))).numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[REAL, FAKE])
    return {
        "video_count": int(len(labels)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_counts": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, default=Path("dataset/original_sequences/youtube/c40/videos"))
    parser.add_argument("--fake-dir", type=Path, default=Path("dataset/manipulated_sequences/Deepfakes/c40/videos"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    torch.set_default_device("cpu")

    records = discover_records(args.real_dir, args.fake_dir)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, global_pool="avg")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    face_model = Path(__file__).resolve().parent / "models" / "blaze_face_short_range.tflite"
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model)),
        min_detection_confidence=0.3,
    )
    detector = vision.FaceDetector.create_from_options(options)

    cache_path = args.output_dir / "face_features.npz"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_signature = [
        f"{record.path}|{Path(record.path).stat().st_size}|"
        f"{Path(record.path).stat().st_mtime_ns}"
        for record in records
    ]
    record_by_path = {record.path: record for record in records}
    features = None

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                "source_signature" in cached.files
                and cached["source_signature"].tolist() == source_signature
            ):
                features = cached["features"]
                kept_labels = cached["labels"]
                kept_groups = cached["groups"]
                cached_paths = cached["paths"].tolist()
                kept_records = [record_by_path[path] for path in cached_paths]
                print(f"Using cached face features: {cache_path}")

    if features is None:
        feature_rows: list[np.ndarray] = []
        kept_records: list[VideoRecord] = []

        for record in records:
            feature = extract_video_feature(
                Path(record.path), model, transform, detector, args.frames_per_video
            )
            if feature is not None:
                feature_rows.append(feature)
                kept_records.append(record)

        if not feature_rows:
            detector.close()
            raise RuntimeError("No usable face crops were found in the dataset.")

        detector.close()
        features = np.stack(feature_rows)
        kept_labels = np.array(
            [record.label for record in kept_records], dtype=np.int64
        )
        kept_groups = np.array([record.group for record in kept_records])

        np.savez_compressed(
            cache_path,
            features=features,
            labels=kept_labels,
            paths=np.array([record.path for record in kept_records]),
            groups=kept_groups,
            source_signature=np.array(source_signature),
        )
        print(f"Saved face feature cache: {cache_path}")
    else:
        detector.close()

    if len(np.unique(kept_labels)) != 2:
        raise ValueError("Need both REAL and FAKE videos after face detection.")

    fold_count = min(5, len(np.unique(kept_groups)))
    if fold_count < 2:
        raise ValueError("Need at least two identity groups for validation.")

    splitter = GroupKFold(n_splits=fold_count)
    oof_scores = np.full(len(kept_labels), np.nan, dtype=np.float64)
    folds = []

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(features, kept_labels, kept_groups), start=1
    ):
        if (
            len(np.unique(kept_labels[train_idx])) != 2
            or len(np.unique(kept_labels[val_idx])) != 2
        ):
            raise ValueError(
                f"Fold {fold_number} lacks REAL or FAKE videos. "
                "Check face detection and identity grouping."
            )

        fold_model = train_classifier(
            features[train_idx],
            kept_labels[train_idx],
            args.epochs,
            args.batch_size,
            args.seed + fold_number,
        )
        with torch.inference_mode():
            scores = torch.sigmoid(
                fold_model(torch.from_numpy(features[val_idx]))
            ).numpy()
        oof_scores[val_idx] = scores

        folds.append({
            "fold": fold_number,
            "test_groups": sorted(set(kept_groups[val_idx])),
            "test_videos": [
                Path(kept_records[i].path).name for i in val_idx
            ],
        })

    if np.isnan(oof_scores).any():
        raise RuntimeError("Some videos did not receive a validation prediction.")

    oof_predictions = (oof_scores >= 0.5).astype(np.int64)
    matrix = confusion_matrix(kept_labels, oof_predictions, labels=[REAL, FAKE])

    # Train a separate final classifier on all cached videos.
    classifier = train_classifier(
        features, kept_labels, args.epochs, args.batch_size, args.seed
    )
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "input_dim": int(features.shape[1]),
            "label_mapping": {"REAL": REAL, "FAKE": FAKE},
            "feature_model": MODEL_NAME,
        },
        args.output_dir / "classifier.pt",
    )

    report = {
        "feature_model": MODEL_NAME,
        "labels": {"REAL": REAL, "FAKE": FAKE},
        "validation_method": "5-fold GroupKFold by identity group",
        "video_count": int(len(kept_labels)),
        "group_count": int(len(np.unique(kept_groups))),
        "video_level_auc": float(roc_auc_score(kept_labels, oof_scores)),
        "video_level_balanced_accuracy": float(
            balanced_accuracy_score(kept_labels, oof_predictions)
        ),
        "confusion_counts": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "folds": folds,
        "out_of_fold_predictions": [
            {
                "video": Path(kept_records[i].path).name,
                "group": str(kept_groups[i]),
                "true_label": int(kept_labels[i]),
                "predicted_label": int(oof_predictions[i]),
                "fake_score": float(oof_scores[i]),
            }
            for i in range(len(kept_labels))
        ],
        "records": [asdict(record) for record in kept_records],
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
