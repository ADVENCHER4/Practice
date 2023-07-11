from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

im = cv2.imread('Images/people.png')
results = model.predict(im)
print(results[0].names)
cam = cv2.VideoCapture(0) # 0 - labtop webcam, 1 - external cam

while True:
    ret, frame = cam.read()

    results = model.predict(source=frame)

    result = results[0].numpy()

    if len(result):
        for i in range(len(results[0])):
            print(i)

            boxes = results[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

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
                result.names[(int(clsID))] + " " + str(round(conf, 3)) + "%",
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



