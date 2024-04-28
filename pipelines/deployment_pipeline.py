import numpy as numpy
import pandas as pd 

from steps.input.get_lines import get_lines
from steps.input.get_words import get_words

# get_lines('steps/mal2.jpg')
def deployment_pipeline(image_path):
    line_boxes = get_lines(image_path)
    get_words(image_path,line_boxes[0])