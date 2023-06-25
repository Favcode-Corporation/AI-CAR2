import torch
import cv2
from time import time
import numpy as np
import os
import gen_annotation

gen_annotation.create_dir_annotation()

# Model
model = torch.hub.load('yolov5', 'custom', path='AI-CAR2-MODEL.pt', source='local', force_reload=True)

cap = cv2.VideoCapture('estrada.mp4')
larguraCap, alturaCap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))
        
    if not ret:
        break

    start_time = time()
    results = model(frame)
    
    end_time = time()
    fps = 1/np.round(end_time - start_time, 2)

    labels = []
    bboxs = []
    
    for color, (_, result) in  zip(colors,  results.pandas().xyxy[0].iterrows()):
        confidence = result['confidence']
        if confidence > 0.8:
            tl = (int(result['xmin']), int(result['ymin']))
            br = (int(result['xmax']), int(result['ymax']))
            labels.append(result['name'])

            print(f"Detectado: {result['name']} C: {confidence}")

            # bbox = [x_min, y_min, largura, altura]
            bboxs.append([tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]])

    if len(labels) > 0:
        _, result = gen_annotation.add(image=frame, labels=labels, bboxs=bboxs, auto_commit=True)
        print(result)

    frame = cv2.putText(frame, "FPS: {:.2f}".format(fps) , (10, 50) , cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow('preview', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()