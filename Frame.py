import cv2
from time import time

cap = cv2.VideoCapture('estrada.mp4')

frame_count = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count > 377 and (frame_count % 80) == 0:
        cv2.imwrite('dataset/frame_' + str(frame_count/80) + '.png', frame)
        cv2.imshow('preview', frame)

    frame_count = frame_count + 1


    cv2.waitKey(1)

    print(frame_count % 100)

cap.release()