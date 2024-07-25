import cv2
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10n')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=640, conf=0.25)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv10 Real-time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()