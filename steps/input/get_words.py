import os

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def get_words(image_path,line):
    img = Image.open(image_path)
    x,y,w,h = line
    plt.imshow(np.array(img)[x:x+w,y+h])