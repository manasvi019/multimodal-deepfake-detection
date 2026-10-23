from preprocessing.video_reader import read_video_info

video_path = "test_videos/sample3.mp4"

info = read_video_info(video_path)

print(info)