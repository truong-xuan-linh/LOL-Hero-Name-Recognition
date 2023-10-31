import os
import cv2
import json
import gdown
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import preprocess_input

from ultralytics import YOLO

VGG_MODEL_DRIVE="https://drive.google.com/file/d/1-0lP3P0L3ehbtGOccPXkzl6PZ5YrJXqw/view?usp=sharing"
VGG_MODEL_DIR="./storage/vgg16_prc.h5"
CLASSES_DIR="./storage/classes.json"

YOLO_DRIVE="https://drive.google.com/file/d/1wHDCrs_ZKmi6NNgyQ5bnQl67wfekfcYa/view?usp=sharing"
YOLO_DIR="./storage/yolov8.pt"

class VGG16:
    def __init__(self) -> None:
        if not os.path.isfile(VGG_MODEL_DIR):
            gdown.download(url=VGG_MODEL_DRIVE, output=VGG_MODEL_DIR, quiet=False, fuzzy=True)
            
        self.model = tf.keras.models.load_model(VGG_MODEL_DIR)
        self.model.trainable = False
        self.classes = {y: x for x, y in json.load(open(CLASSES_DIR)).items()}
        
        self.extract_model = Model(inputs=self.model.input, outputs=self.model.get_layer("dense_12").output)
        
    def predict(self, image):
        
        image = cv2.resize(image, (224,224))
        image = preprocess_input(image)
        input = np.expand_dims(image, axis=0)
        idx = np.argmax(self.model.predict(input, verbose=0)[0])
        
        return self.classes[idx]
        
    def get_feature(self, image):
        
        image = cv2.resize(image, (224,224))
        image = preprocess_input(image)
        input = np.expand_dims(image, axis=0)
        feature = self.extract_model.predict(input, verbose=0)[0]
        
        return feature

class ChampitionDetector:
    def __init__(self) -> None:
        if not os.path.isfile(YOLO_DIR):
            gdown.download(url=YOLO_DRIVE, output=YOLO_DIR, quiet=False, fuzzy=True)
            
        self.model = YOLO(YOLO_DIR)
    
    def predict(self, image_dir, conf=0.7):
        image = Image.open(image_dir)
        yolo_rs = self.model.predict(image_dir, conf=conf)[0]
        boxes = yolo_rs.boxes.data.tolist()
        
        result = {
            "images": [],
            "left_champion": image,
        }
        x_left = image.size[0]
        for box in boxes:
            x_lt, y_lt, x_rb, y_rb, _, _ = box
            crop_image = image.crop((x_lt, y_lt, x_rb, y_rb))
            result["images"].append(crop_image)
            if x_lt < x_left:
                x_left = x_lt
                result["left_champion"] = crop_image
        
        return result