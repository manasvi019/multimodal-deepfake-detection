
import librosa

def load_audio(audio_path):
    """Load audio and return its signal and sample rate."""

    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    return audio, sample_rate
