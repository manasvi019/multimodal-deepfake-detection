import json
from pathlib import Path

import numpy as np
import torch

from models.aasist_model import Model


class AudioAnalyzer:
    def __init__(self):
        project_dir = Path(__file__).resolve().parents[1]
        checkpoint_path = project_dir / "models" / "audio_aasist.pth"
        config_path = project_dir / "models" / "aasist_config.conf"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Audio checkpoint not found: {checkpoint_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Audio config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)["model_config"]

        self.target_samples = config["nb_samp"]
        self.model = Model(config).to("cpu")

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_samples(
        self,
        audio_samples: np.ndarray | None,
        sample_rate: int = 16000,
    ) -> float | None:
        if audio_samples is None:
            return None
        if sample_rate != 16000:
            raise ValueError("Expected audio sampled at 16000 Hz.")

        audio = np.asarray(audio_samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Expected mono audio as a one-dimensional array.")
        if audio.size == 0:
            return None
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid values.")

        if audio.size < self.target_samples:
            repeats = int(np.ceil(self.target_samples / audio.size))
            audio = np.tile(audio, repeats)

        audio = audio[:self.target_samples]
        waveform = torch.from_numpy(audio.copy()).unsqueeze(0)

        _, logits = self.model(waveform)
        probabilities = torch.softmax(logits, dim=1)[0]

        # AASIST class 0 is spoof; this is the model's fake score.
        return float(probabilities[0].item())