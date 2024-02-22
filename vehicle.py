import cv2
import numpy as np

#WebCam/Video
cap = cv2.VideoCapture('video.mp4')
# cap = cv2.VideoCapture(0)

#Algo
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

#Line
count_line_position = 550
offset = 6
counter = 0

#Min Height and Width
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
    countour, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 255, 0), 3)

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
            cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 0, 255))
            detect.remove((x, y))
        
    cv2.putText(frame, "VEHICLE COUNT: " + str(counter), (450, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 100, 50), 4)

    cv2.imshow('output', frame)
    cv2.imshow('contours', dilated)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()