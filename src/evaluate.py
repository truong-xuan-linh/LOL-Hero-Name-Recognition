import os
from src.query import Find

class Evaluate:
    def __init__(self) -> None:
        self.find = Find()
    
    def eval(self, image_path, save_path):
        
        images_name = os.listdir(image_path)
        for image_name in images_name:
            pre = self.find.find_best(f"{image_path}/{image_name}")
            with open(f"{save_path}/output.txt", "a") as f:
                f.write(f"{image_name}\t{pre}\n")
                
    @staticmethod
    def accuracy(ground_truth, predict):
        with open(ground_truth, "r") as f:
            grounds = f.readlines()
            
        grounds = [g.replace("\n", "") for g in grounds]
        grounds_dict = {}
        for g in grounds:
            k, v = g.split("\t")
            grounds_dict[k] = v
            
        with open(predict, "r") as f:
            predicts = f.readlines()
        predicts = [p.replace("\n", "") for p in predicts]
        predicts_dict = {}
        for p in predicts:
            k, v = p.split("\t")
            predicts_dict[k] = v
        
        count = 0
        for k, v in predicts_dict.items():
            if v == grounds_dict[k]:
                count+=1
            else:
                print(k, grounds_dict[k], v)
        
        return count/predicts_dict.__len__()