
from preprocessing.face_detector import detect_faces

image_path = "extracted_frames/frame_000120.jpg"

faces = detect_faces(image_path)

print("Faces detected:", len(faces))
print("Detection details:", faces)
