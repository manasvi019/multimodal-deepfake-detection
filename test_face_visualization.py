
import cv2
from preprocessing.face_detector import detect_faces

image_path = "extracted_frames/frame_000120.jpg"
image = cv2.imread(image_path)

faces = detect_faces(image_path)

for face in faces:
    x = face["x"]
    y = face["y"]
    w = face["width"]
    h = face["height"]

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"Face: {face['score']:.2f}"
    cv2.putText(
        image, label, (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

cv2.imwrite("face_detection_result.jpg", image)

print("Faces detected:", len(faces))
print("Saved visualization: face_detection_result.jpg")
