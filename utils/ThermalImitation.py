from ultralytics import YOLO
import cv2
import numpy as np
import sys

sys.path.append(r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai")
from sort import Sort
from utils.Thermalimaging import generateColourMap
import time

model = YOLO("runs/detect/train3/weights/best.pt")

tracker = Sort(max_age=40, min_hits=3, iou_threshold=0.3)

class_names = model.names

path1 = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai\Deer feeding around my camera trap with the infrared.mp4"
path2 = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai\Animal.mp4"

colorMap = generateColourMap()
cap = cv2.VideoCapture(path2)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    # frame = cv2.resize(frame,(480, 360))
    results = model.predict(source=frame, conf=0.75, iou=0.5, verbose=False)

    detections = []
    labels = []
    bboxes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            roi = frame[y1:y2, x1:x2]

            # green_overlay = np.full(roi.shape, (0, 255, 0), dtype=np.uint8)

            normed = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            gray = cv2.cvtColor(normed, cv2.COLOR_BGR2GRAY)
            cl1 = clahe.apply(gray)
            nor = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

            colorized_img = cv2.LUT(nor, colorMap)

            blended = cv2.addWeighted(roi, 0.4, colorized_img, 0.6, 0)
            frame[y1:y2, x1:x2] = blended

            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = class_names[cls]
            detections.append([x1, y1, x2, y2, conf])
            labels.append(label)
            bboxes.append((x1, x2, y1, y2))

    detections = np.array(detections)
    if detections.shape[0] == 0:
        detections = np.empty((0, 5))

    tracks = tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
