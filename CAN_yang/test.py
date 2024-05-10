import os
import cv2
import argparse
import torch
import json
import pickle as pkl
from tqdm import tqdm
import time
from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model import Inference
from dataset import Words
import numpy as np

# image_path = r"D:\ZJU\research\sketch\CAN\datasets\CROHME\14_test_images.pkl"
config_file = 'config.yaml'

# with open(image_path, 'rb') as f:
#     images = pkl.load(f)

image = cv2.imread("./images/2.png", cv2.IMREAD_GRAYSCALE)
_, image = cv2.threshold(image, 1, 255, cv2.THRESH_OTSU)
H, W = image.shape[0], image.shape[1]
# print(H, W)
image = cv2.resize(image, (W//5, H//5))

# image = images['18_em_10']
# cv_image = np.uint(image)
# cv2.imshow("image", image)


img = torch.Tensor(255-image) / 255
img = img.unsqueeze(0).unsqueeze(0)
img = img.cuda()
params = load_config(config_file)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
words = Words('datasets/CROHME/words_dict.txt')
params['word_num'] = len(words)

model = Inference(params, draw_map=False)
model = model.to(device)
load_checkpoint(model, None, params['checkpoint'])
model.eval()
with torch.no_grad():
    probs, _ = model(img, None, None)
    prediction = words.decode(probs)
print(prediction)
cv2.waitKey(0)