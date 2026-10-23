from preprocessing.audio_extractor import extract_audio

video_path = "test_videos/sample.mp4"
output_path = "extracted_audio/sample.wav"

result = extract_audio(video_path, output_path)

print(f"Audio saved to: {result}")