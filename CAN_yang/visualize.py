import torch
import time
import pickle as pkl
import cv2
import numpy as np

image_path = r"D:\ZJU\research\sketch\CAN\datasets\CROHME\14_test_images.pkl"

with open(image_path, 'rb') as f:
    images = pkl.load(f)

image = images['18_em_0']
image = np.uint8(image)
print(image)
cv2.imshow("image", image)
cv2.waitKey(0)



