import os
import numpy as np
import pandas as pd

from src.model import VGG16
from src.model import ChampitionDetector
from src.utils import Utils


class Find:
    def __init__(self) -> None:
        # self.embedds = {}
        # list_embedd = os.listdir(embedd_folder)
        # for embedd in list_embedd:
        #     self.embedds[embedd] = np.load(f"{embedd_folder}/{embedd}")
        
        self.vgg16 = VGG16()
        self.detector = ChampitionDetector()

    def find_best(self, image_dir):
        detect_result = self.detector.predict(image_dir)
        left_champion = detect_result["left_champion"]
        if not left_champion:
            return None
        
        result = self.vgg16.predict(Utils.bound_image(np.array(left_champion.convert("RGB"))))
        return result
    
    def find_top_k(self, image_dir):
        detect_result = self.detector.predict(image_dir)
        left_champion = detect_result["left_champion"]
        if not left_champion:
            return None
        
        embedding = self.vgg16.get_feature(Utils.bound_image(np.array(left_champion.convert("RGB"))))
        results = []
        for key, value in self.embedds.items():
            similarity = Utils.find_cosine(embedding, value)
            results.append({
                "name": ".".join(key.split(".")[:-1]),
                "similarity": similarity
            }
            )
        
        df = pd.DataFrame(results)
        df = df.sort_values("similarity")
        return df