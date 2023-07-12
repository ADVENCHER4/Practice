from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture(0)  # 0 - laptop webcam, 1 - external cam
detObj = [0, 2, 3, 5, 7]

while True:
    ret, frame = cam.read()
    results = model.predict(source=frame)
    result = results[0].numpy()
    if len(result):
        for i in range(len(results[0])):
            boxes = results[0].boxes
            box = boxes[i]  # returns one box
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            if clsID in detObj:
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (255, 0, 0),
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    f'{result.names[clsID]} {round(conf, 3)}%',
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()



