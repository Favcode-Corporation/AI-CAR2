import torch
import cv2
from time import time
import numpy as np
import os


# Model
model = torch.hub.load('/yolov5', 'custom', path='/AI-CAR2-MODEL.pt', source='local', force_reload=True)

cap = cv2.VideoCapture('estrada.mp4')
larguraCap, alturaCap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

while True:
    ret, frame = cap.read()
        
    start_time = time()
    results = model(frame)
    
    end_time = time()
    fps = 1/np.round(end_time - start_time, 2)
        
    frame = cv2.putText(frame, "FPS: {:.2f}".format(fps) , (10, 50) , cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    
    for color, (_, result) in  zip(colors,  results.pandas().xyxy[0].iterrows()):
        confidence = result['confidence']
        if confidence > 0.8:
            tl = (int(result['xmin']), int(result['ymin']))
            br = (int(result['xmax']), int(result['ymax']))
            label = result['name']

            #distance calculator
            frame = cv2.line(frame, (int(larguraCap/2), (alturaCap-5)),
                            (int((tl[0]+br[0])/2), br[1]), color, 5)
            
            distancia = ((alturaCap) -br[1])*0.0256


            frame = cv2.line(frame, tl, br, color, 5)
            frame = cv2.putText(frame, label + ' C: {:.2f}'.format(confidence), tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            localTxt = tl[0], br[1]
            frame = cv2.putText(frame, 'D: {:.1f}m'.format(distancia), localTxt, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

    cv2.imshow('preview', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()