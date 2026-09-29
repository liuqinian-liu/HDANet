import os
import os.path
import cv2
import numpy as np


path= './'+str(16)+'/'
files_list = os.listdir(path)
root = './'
roots = 'dataset/labels/'
with open("16.txt",'a',encoding='utf-8') as filetext:
        for file in files_list:
            filetext.write(file + "\n")