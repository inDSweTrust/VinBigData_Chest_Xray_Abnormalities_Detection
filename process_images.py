import os

from PIL import Image
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from skimage import exposure
from pathlib import Path

from config import FlagsDet, flags_yolo

import matplotlib.pyplot as plt

flags = FlagsDet().update(flags_yolo)
out_size = flags.dim
print(f"Image Size: {out_size}")
inputdir = Path(flags.inputdir)

def read_xray(path, voi_lut = True, fix_monochrome = True, clahe=False, hist=False):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        image = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        image = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image
        
    intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else 0.0
    slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1.0
        
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
            
    image += np.int16(intercept)        
        
    image = image - np.min(image)

    # Hist normalization
    if hist:
        image = exposure.equalize_hist(image)
    # CLAHE normalization
    if clahe:
        image = exposure.equalize_adapthist(image/np.max(image))
    else:
        image = image / np.max(image)
        
    image = (image * 255).astype(np.uint8)

    return image

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im


def worker(filepath, save_dir, load_dir, out_size=out_size):
    xray = read_xray(load_dir + filepath, clahe=False, hist=False)
    im = resize(xray, size=out_size, keep_ratio=False)
    im.save(save_dir + filepath.replace('dicom', 'png'))
    return xray.shape

def worker_train(filepath, worker_fn):
    shape = worker_fn(filepath)
    return (filepath.replace('.dicom', ''), *shape[:2])

# EXAMPLE
img = read_xray(inputdir / 'train/1df9d68d3a8352f4fecbc3895ebd80fa.dicom')
# img = exposure.equalize_adapthist(img/np.max(img))
plt.figure(figsize = (12,12))
plt.imshow(img, 'gray')

# TRAIN AND TEST
image_id = []
dim0 = []
dim1 = []

DIR_INPUT = inputdir

for split in ['train', 'test']:
    load_dir = f'{DIR_INPUT}/{split}/'
    save_dir = f'{DIR_INPUT}/images_{out_size}/{split}/'

    os.makedirs(save_dir, exist_ok=True)
    
    worker_fn = partial(worker, save_dir=save_dir, load_dir=load_dir)
    cur_worker_fn = partial(worker_train, worker_fn=worker_fn) if split == 'train' else worker_fn
    
    with Pool(12) as p:
        results = p.map(cur_worker_fn, os.listdir(load_dir))
    if split == 'train':
        for img_id, *cur_dim in results:
            image_id.append(img_id)
            dim0.append(cur_dim[0])
            dim1.append(cur_dim[1])
            
df = pd.DataFrame.from_dict({'image_id': image_id, 'dim0': dim0, 'dim1': dim1})
df.to_csv(inputdir / 'train_meta.csv', index=False)

# # TEST ONLY
# image_id = []
# dim0 = []
# dim1 = []


# load_dir = f'{DIR_INPUT}/test/'
# save_dir = f'{DIR_INPUT}/images_{out_size}/test/'

# os.makedirs(save_dir, exist_ok=True)

# worker_fn = partial(worker, save_dir=save_dir, load_dir=load_dir)
# cur_worker_fn = partial(worker_train, worker_fn=worker_fn)

# with Pool(12) as p:
#     results = p.map(cur_worker_fn, os.listdir(load_dir))
    
# for img_id, *cur_dim in results:
#     image_id.append(img_id)
#     dim0.append(cur_dim[0])
#     dim1.append(cur_dim[1])
        

# df = pd.DataFrame.from_dict({'image_id': image_id, 'dim0': dim0, 'dim1': dim1})
# df.to_csv('test_meta.csv', index=False)
