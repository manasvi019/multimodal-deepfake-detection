from preprocessing.frame_extractor import extract_frames

video_path = "test_videos/sample2.mp4"
output_dir = "extracted_frames"

count = extract_frames(video_path, output_dir)

print(f"Frames extracted: {count}")