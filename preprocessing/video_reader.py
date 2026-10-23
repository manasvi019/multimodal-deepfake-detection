
import cv2


def read_video_info(video_path):  # function
    """Read basic information about a video file."""
    video = cv2.VideoCapture(str(video_path)) #open video for reading nad converting path into string
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    #basic video properties
    fps = video.get(cv2.CAP_PROP_FPS) #how many frames the video displays each sec
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release() #close the video file after reading its properties

    duration = total_frames / fps if fps > 0 else 0 #calculate the duration of the video in seconds

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }
