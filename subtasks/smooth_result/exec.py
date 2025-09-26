import cv2
import numpy as np

import os, sys
sys.path.append(os.getcwd())

def main():
    assets_root = "assets/intervention"
    
    for filename in os.listdir(assets_root):
        filepath = f"{assets_root}/{filename}"
        
        if filename.endswith("GT.jpg") or filename.endswith("smooth.jpg"):
            continue
        else:
            img = cv2.imread(filepath)
            blurred = cv2.medianBlur(img, 5)
            # print(f"{filepath[:-4]}_smooth.jpg")
            cv2.imwrite(f'{filepath[:-4]}_smooth.jpg', blurred)

if __name__ == '__main__':
    main()