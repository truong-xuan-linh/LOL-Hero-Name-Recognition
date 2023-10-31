import argparse
from src.evaluate import Evaluate
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_folder', required=True, help='image folder directory')
  parser.add_argument('--save_folder', required=False, default="./storage", help='output.txt save directory')
  
  args = parser.parse_args()
  image_folder = args.image_folder
  save_folder = args.save_folder
  
  print("PROCESSING...")
  evaluate = Evaluate()
  evaluate.eval(image_path=image_folder, save_path=save_folder)
  print(f"DONE!!! The result have been saved at {save_folder}/output.txt")
  
if __name__ == '__main__':
    main()