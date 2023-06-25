import torch
import cv2
from time import time
import numpy as np
import gen_annotation
import argparse
from colorama import Fore

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, help='Input Video Path', required=True)
parser.add_argument('--last_annotation_path', type=str, help='Last Annotation Path')
args = parser.parse_args()

if args.last_annotation_path == '':
    gen_annotation.create_dir_annotation()
else:
    last_image = gen_annotation.load_annotation(args.last_annotation_path)

# Model
model = torch.hub.load('yolov5', 'custom', path='AI-CAR2-MODEL.pt', source='local', force_reload=True)

cap = cv2.VideoCapture(args.video_path)
larguraCap, alturaCap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

quant_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
stime_video = time()
count_frames = 0
porcent_processamento = ''

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))


    if not ret:
        break

    tempo_atual = (time() - stime_video)/60

    temp_porcent_processamento = '{:.2f}%'.format((count_frames/quant_frames)*100)

    if temp_porcent_processamento != porcent_processamento:
      print(Fore.BLUE + "Processamento: " + temp_porcent_processamento, 'Tempo Atual: {:.2f}min'.format(tempo_atual))
      porcent_processamento = temp_porcent_processamento

    count_frames += 1

    if count_frames > last_image:
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

                print(Fore.GREEN + f"Detectado: {result['name']} C: {confidence}")

                # bbox = [x_min, y_min, largura, altura]
                bboxs.append([tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]])

        if len(labels) > 0:
            _, result = gen_annotation.add(image=frame, labels=labels, bboxs=bboxs, auto_commit=True)
            print(Fore.YELLOW + result + "  FPS: {:.2f}".format(fps))
    else:
        print(Fore.YELLOW + "Image has already been annotated")
    

cap.release()