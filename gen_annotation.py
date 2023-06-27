import json
import os
import cv2
import uuid
from datetime import datetime
import time
import pytz
import warnings

path_dir = 'AI-CAR2-DATASET-ML'

categories = {
    "Caminhao": 1,
    "Carro": 2,
    "Moto": 3,
    "Onibus": 4
}

data_atual = datetime.now(pytz.utc)
fuso_horario = pytz.timezone('UTC')
data_atual_fuso = data_atual.astimezone(fuso_horario)
data_formatada = data_atual_fuso.strftime('%Y-%m-%dT%H:%M:%S%z')

coco_annotations = {
    "info": {
        "year": "2023",
        "version": "2",
        "description": "Created from AI-DATA2 ML",
        "contributor": "Thigos Rodrigues",
        "url": "https://public.roboflow.ai/object-detection/undefined",
        "date_created": data_formatada,
        "last_frame": 0
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://choosealicense.com/licenses/mit/",
            "name": "MIT"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "cars-trucks-bikes-",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "Caminhao",
            "supercategory": "cars-trucks-bikes-"
        },
        {
            "id": 2,
            "name": "Carro",
            "supercategory": "cars-trucks-bikes-"
        },
        {
            "id": 3,
            "name": "Moto",
            "supercategory": "cars-trucks-bikes-"
        },
        {
            "id": 4,
            "name": "Onibus",
            "supercategory": "cars-trucks-bikes-"
        }
    ],
    "images": [],
    "annotations": [],
}

def load_annotation(path):
    global path_dir, coco_annotations
    with open(f'{path}/_annotations.coco.json', "r") as file:
        last_annotation = json.load(file)
        coco_annotations = last_annotation
        path_dir = path
        print(f"Images Loaded: {len(coco_annotations['images'])}", '|' ,f"Annotations Loaded: {len(coco_annotations['annotations'])}")
        time.sleep(1)
        return coco_annotations['info']['last_frame']

def create_dir_annotation():
    global path_dir
    count_dir = 1

    for file in os.listdir('.'):
        if os.path.isdir(file) and path_dir in file:
            count_dir += 1

    try:
        if count_dir == 1:
            os.mkdir(path_dir)
        else:
            path_dir = f'{path_dir}{count_dir}'
            os.mkdir(path_dir)
    except:
        raise

def write_image(image):
    chave = uuid.uuid4()
    file_name = f'{chave}.png'
    sucess = cv2.imwrite(f'{path_dir}/{file_name}', image)

    if not sucess:
        raise Exception('Unable to Save Annotation Image :(')
    else:
        return file_name

def add(image, labels, bboxs, auto_commit, last_frame):
    global coco_annotations

    coco_annotations['info']['last_frame'] = last_frame

    image_id = len(coco_annotations['images'])+1
    image_file_name = write_image(image)
    height, width, _ = image.shape

    data_atual = datetime.now(pytz.utc)
    fuso_horario = pytz.timezone('UTC')
    data_atual_fuso = data_atual.astimezone(fuso_horario)
    data_formatada = data_atual_fuso.strftime('%Y-%m-%dT%H:%M:%S%z')

    coco_annotations["images"].append({
        "id": image_id,
        "license": 1,
        "file_name": image_file_name,
        "height": height,
        "width": width,
        "date_captured": data_formatada
    })


    for label, bbox in zip(labels, bboxs):
        annotation_id = len(coco_annotations['annotations'])+1
        category_id = categories[label]

        coco_annotations["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "segmentation": [],
            "iscrowd": 0
        })

    if auto_commit:
        warnings.warn('AUTO COMMIT IS ENABLED')
        commit(coco_annotations)
        return coco_annotations, 'Annotation Added'
    else:
        return coco_annotations, 'Annotation Added'

def commit(coco_annotations):
    with open(f"{path_dir}/_annotations.coco.json", "w") as file:
        json.dump(coco_annotations, file)