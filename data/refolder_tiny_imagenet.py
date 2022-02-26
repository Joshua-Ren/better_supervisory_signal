import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T

def findallfiles(BASE_PATH):
    for root, ds, fs in os.walk(BASE_PATH):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname
def del_empty_folder(BASE_PATH):
    for root, dirs, files in os.walk(BASE_PATH, topdown=False):
        if not files and not dirs:
            os.rmdir(root)    

# Define main data directory
DATA_DIR = './tiny-imagenet-200/'
TRAIN_DIR = DATA_DIR + 'train/'
VALID_DIR = DATA_DIR + 'val/'

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(VALID_DIR)

# Open and read val annotations text file
fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    image_ = os.path.join(val_img_dir,'images', img)
    if os.path.exists(image_):
        os.rename(image_, os.path.join(newpath, img))
for file in findallfiles(VALID_DIR):
    if file[-3:]=='txt':
        os.remove(file)
del_empty_folder(VALID_DIR)
# -------- remove the image file from the path
for file in findallfiles(TRAIN_DIR):
    if file[-3:]=='txt':
        os.remove(file)
    else:
        os.rename(file, file.replace('/images/','/'))
del_empty_folder(TRAIN_DIR)
