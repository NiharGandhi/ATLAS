import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
# Get the indices of the output layers and convert to list
layer_indices = net.getUnconnectedOutLayers().flatten().tolist()
output_layers = [layer_names[idx - 1] for idx in layer_indices]


# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Colors for different classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# WebCam/Video
cap = cv2.VideoCapture('video.mp4')
# cap = cv2.VideoCapture(0)

# Algo
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Line
count_line_position = 550
offset = 6
counter = 0

# Min Height and Width
min_width = 80
min_height = 80


def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy


detect = []

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    countour, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, count_line_position),
             (1200, count_line_position), (255, 255, 0), 3)

    for (i, c) in enumerate(countour):
        (x, y, w, h) = cv2.boundingRect(c)
        val_counter = (w >= min_width) and (h >= min_height)
        if not val_counter:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
            cv2.line(frame, (25, count_line_position),
                     (1200, count_line_position), (0, 0, 255))
            detect.remove((x, y))

    # Vehicle detection using YOLOv4-tiny
    height, width, channels = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30),
                        cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    cv2.putText(frame, "VEHICLE COUNT: " + str(counter), (450, 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 100, 50), 4)

    cv2.imshow('output', frame)
    cv2.imshow('contours', dilated)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
