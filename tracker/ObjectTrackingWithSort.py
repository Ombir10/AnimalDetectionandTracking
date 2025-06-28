from ultralytics import YOLO
import sys

sys.path.append(r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai")
from sort import Sort
import cv2
import numpy as np
import time
import os

model = YOLO("runs/detect/train3/weights/best.pt")

tracker = Sort(max_age=40, min_hits=3, iou_threshold=0.3)

class_names = model.names

path2 = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai\Animal.mp4"

vid2 = cv2.VideoCapture(path2)

frame_id = 0

# Open output file in MOT format
os.makedirs("track_logs", exist_ok=True)
log_sort = open("track_logs/sort_output.txt", "w")
log_byte = open("track_logs/bytetrack_output.txt", "w")

pth_sort = (
    r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai\utils\sort.yaml"
)


def capture(cap, log_file, frame_id):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        results = model.predict(source=frame, conf=0.75, iou=0.5, verbose=False)

        frame_id += 1

        detections = []
        labels = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = class_names[cls]
                detections.append([x1, y1, x2, y2, conf])
                labels.append(label)

        detections = np.array(detections)
        if detections.shape[0] == 0:
            detections = np.empty((0, 5))

        tracks = tracker.update(detections)

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            w = x2 - x1
            h = y2 - y1
            confidence = 1  # SORT doesn’t give confidence per track

            log_file.write(
                f"{frame_id},{int(track_id)},{x1},{y1},{w},{h},{confidence},-1,-1,-1\n"
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    log_file.close()
    cv2.destroyAllWindows()


def log_track(cap, frame_id, log_file):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        results = model.track(
            frame, tracker="bytetrack.yaml", persist=True, verbose=False
        )

        if results[0].boxes.id is None:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            log_file.write(
                f"{frame_id},{int(track_id)},{x1},{y1},{w},{h},{conf:.2f},-1,-1,-1\n"
            )

        cv2.imshow("ByteTrack Tracking", results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()


result = capture(vid2, log_sort, frame_id)
# results = log_track(vid2, frame_id, log_byte)
