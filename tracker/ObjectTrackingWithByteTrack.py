from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")
class_names = model.names

# Load video
pth = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai\Animal.mp4"
vid = cv2.VideoCapture(pth)


def log_track(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, tracker="bytetrack.yaml", persist=True, verbose=False
        )

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, conf, cls_id in zip(boxes, ids, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label_name = class_names[cls_id] if cls_id in class_names else str(cls_id)
            label = f"{label_name} Conf:{conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("ByteTrack Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run tracking
log_track(vid)
