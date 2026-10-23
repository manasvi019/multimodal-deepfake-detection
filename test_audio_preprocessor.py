
from preprocessing.audio_preprocessor import load_audio

audio, sample_rate = load_audio("extracted_audio/sample.wav")

print("Sample rate:", sample_rate)
print("Audio samples:", len(audio))
print("Duration:", round(len(audio) / sample_rate, 2), "seconds")
