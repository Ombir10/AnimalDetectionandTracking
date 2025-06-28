import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Load YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")

# Initialize box annotator
box_annotator = sv.BoxAnnotator(thickness=2)

# Load video
pth = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai\Animal.mp4"
cap = cv2.VideoCapture(pth)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Track and annotate
for result in model.track(source=pth, show=True, stream=True):
    frame = result.orig_img

    # Try from_ultralytics if you have the latest supervision version
    try:
        detections = sv.Detections.from_ultralytics(result)
    except AttributeError:
        detections = sv.Detections.from_yolov8(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        labels = [
            f"ID: {tracker_id} Conf: {confidence:.2f}"
            for tracker_id, confidence in zip(
                detections.tracker_id, detections.confidence
            )
        ]

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
