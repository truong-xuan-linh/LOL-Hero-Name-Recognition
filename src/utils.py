import cv2
import numpy as np

class Utils:

    @staticmethod
    def bound_image(origin_image):
        h, w, _ = origin_image.shape
        bounder = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(bounder, (int(w//2), int(h//2)), (int(w//2.3), int(h//2.3)), 0, 0, 360, (1, 1, 1), -1)
        
        new_image = origin_image*bounder
        return new_image
    
    @staticmethod
    def find_cosine(A, B):
        cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
        return cosine