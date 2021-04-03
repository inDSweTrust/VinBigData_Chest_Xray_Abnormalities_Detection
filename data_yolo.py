import shutil, os
import yaml
import json
import random

import numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
from pathlib import Path

from config import FlagsDet, flags_yolo

from os import listdir
from os.path import isfile, join

# Config
flags = FlagsDet().update(flags_yolo)
dim = flags.dim
fold = flags.fold
use_class14 = flags.use_class14
print(f"IMG: {dim}, fold {fold}, use_class14: {use_class14}")
# Classes
classes = ['Aortic enlargement',
            'Atelectasis',
            'Calcification',
            'Cardiomegaly',
            'Consolidation',
            'ILD',
            'Infiltration',
            'Lung Opacity',
            'Nodule/Mass',
            'Other lesion',
            'Pleural effusion',
            'Pleural thickening',
            'Pneumothorax',
            'Pulmonary fibrosis']
if use_class14:
    classes.append('No finding')
    
# Directories
inputdir = Path(flags.inputdir)
imgdir = Path(inputdir / f'images_{dim}')
yolodir = Path(flags.yolodir)
outdir = Path(flags.outdir)
os.makedirs(str(outdir), exist_ok=True)

#Based on https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-train
def create_norm_df_folds(inputdir, dim=512, use_class14=True, use_nih=False, fold=4):
    # Create Normalized train_df
    # Concatenate original dataset with external dataset - specify external dataset path
    if use_nih:
        vbd_df = pd.read_csv(inputdir / 'train_wh.csv')
        vbd_df.fillna(0, inplace=True)
        nih_df = pd.read_csv(inputdir / 'train_wh_nih.csv')
  
        train_df = pd.concat([vbd_df, nih_df])
        train_df.fillna(0, inplace=True)
        train_df.reset_index(drop=True, inplace=True)

        train_df.loc[0:67914,'image_path'] = flags.inputdir + f'/images_{dim}/train/' + train_df.image_id+('.png' if dim!='original' else '.jpg')
        train_df.loc[67914:69721,'image_path'] = flags.inputdir + f'/NIH/images_{dim}/'+train_df.image_id+('.png')

    else:
        train_df = pd.read_csv(inputdir / 'train_wh_wbf.csv') # Train with WBF
        train_df.fillna(0, inplace=True)
    
        train_df['image_path'] = flags.inputdir + f'/images_{dim}/train/' + train_df.image_id+('.png' if dim!='original' else '.jpg')
    
    if use_class14 == False:
        train_df = train_df[train_df.class_id!=14].reset_index(drop = True)
    
    
    train_df['x_min'] = train_df.apply(lambda row: (row.x_min)/row.width, axis =1)
    train_df['y_min'] = train_df.apply(lambda row: (row.y_min)/row.height, axis =1)
    
    train_df['x_max'] = train_df.apply(lambda row: (row.x_max)/row.width, axis =1)
    train_df['y_max'] = train_df.apply(lambda row: (row.y_max)/row.height, axis =1)
    train_df.loc[train_df["class_id"] == 14, ['x_max', 'y_max']] = 1.0
    
    train_df['x_mid'] = train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
    train_df['y_mid'] = train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)
    
    train_df['w'] = train_df.apply(lambda row: (row.x_max-row.x_min), axis =1)
    train_df['h'] = train_df.apply(lambda row: (row.y_max-row.y_min), axis =1)
    
    train_df['area'] = train_df['w']*train_df['h']
    
    # Fold split
    gkf  = GroupKFold(n_splits = 5)
    train_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups = train_df.image_id.tolist())):
        train_df.loc[val_idx, 'fold'] = fold
    train_df
    
    train_df.to_csv(yolodir / 'df_norm_folds.csv', index=False)
    
    return train_df

# create_norm_df_folds(inputdir, dim=512, use_class14=True, use_nih=False, fold=4)

def change_path_norm_df_folds(dim, dir=yolodir):
    df = pd.read_csv(dir / 'df_norm_folds.csv')
    for i, path in enumerate(df['image_path'].values):
        split_path = path.split('/')
        split_path[6] = f'images_{dim}'
        df.loc[i, 'image_path'] = '/'.join(split_path)
    
    df.to_csv(dir / 'df_norm_folds.csv', index=False)
    
# change_path_norm_df_folds(dim=512, dir=yolodir)

def create_single_cls_df(dir=yolodir, use_class14=True, cls=2):
    df = pd.read_csv(dir / 'df_norm_folds.csv')
    if use_class14:
        df = df[(df.class_id == cls) | (df.class_id == 14)]
    else:
        df = df[df.class_id == cls]
    df.to_csv(dir / f'df_norm_folds_{classes[cls]}.csv', index=False)

# create_single_cls_df(dir=yolodir, use_class14=True, cls=2)

def annotations_coco_format(fold=fold, dim=512, val300=False):
    df = pd.read_csv(yolodir / 'df_norm_folds.csv')
    cat_df = df[['class_id', 'class_name']].drop_duplicates()
    df = df[df.fold == fold]
    
    if val300:
        val_300 = []
        with open(join(yolodir, 'yolo_files/val300.txt'), 'r') as f:
            for path in f:
                split1 = path.split('/')[9]
                imgid = split1.split('.')[0]
                val_300.append(imgid)
        
        df = df[df['image_id'].isin(val_300)]
    
    k = 0
    images = []
    categories = []
    annotations = []            
    for index, row in df.iterrows():
        images.append({
            "id": row['image_id'],
            "width": dim,
            "height": dim,
            "file_name":row["image_path"]
        })
        annotations.append({
            "id": k,
            "image_id": row['image_id'],
            "category_id": row["class_id"],
            "bbox":[row["x_min"]*dim,
                    row["y_min"]*dim,
                    row["w"]*dim,
                    row["h"]*dim],
            "segmentation": [],
            "ignore": 0,
            "area": row['area']*dim,
            "iscrowd": 0,
        })
        k += 1
    
    for index, row in cat_df.sort_values(by=['class_id']).iterrows():
        categories.append({'id': row['class_id'], 'name': row['class_name']})
        
    coco_dict = {}
    coco_dict['images'] = images
    coco_dict['categories'] = categories
    coco_dict['annotations'] = annotations
    
    out = Path(yolodir / 'yolo_files')
    os.makedirs(str(out), exist_ok=True)
    if val300:
        with open(out / f'gt_coco_format_fold{fold}_val300.json', 'w') as f:
            f.write(json.dumps(coco_dict))
            f.flush()
    else:
        with open(out / f'gt_coco_format_fold{fold}.json', 'w') as f:
            f.write(json.dumps(coco_dict))
            f.flush()
           
    return coco_dict

# annotations_coco_format(fold=fold, dim=dim, val300=False)


def make_yolo_train_files(dir=yolodir, fold=fold, single_cls=True, cls=2):
    if single_cls:
        train_df = pd.read_csv(dir / f'df_norm_folds_{classes[cls]}.csv')
    else:
        train_df = pd.read_csv(dir / 'df_norm_folds.csv')
    train_files = []
    val_files   = []
    val_files += list(train_df[train_df.fold==fold].image_path.unique())
    train_files += list(train_df[train_df.fold!=fold].image_path.unique())
    print(len(train_files), len(val_files))
    
    out = Path(dir / 'yolo_files')
    os.makedirs(str(out), exist_ok=True)

    with open(join(out , 'train.txt'), 'w') as f:
        for path in glob(flags.yolodir + '/input/images/train/*'):
            f.write(path+'\n')
                
    with open(join(out, 'val.txt'), 'w') as f:
        for path in glob(flags.yolodir + '/input/images/val/*'):
            f.write(path+'\n')
    
    data = dict(
        train =  join(out , 'train.txt') ,
        val   =  join(out , 'val.txt' ),
        nc    = len(classes),
        names = classes
        )
    
    with open(join(out ,'vinbigdata.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
    f = open(join(out ,'vinbigdata.yaml'), 'r')
    print('\nyaml:')
    print(f.read())
    
    yoloindir = Path(yolodir / 'input')
    # Make txt files containing normalized labels (cat, bbox)
    os.makedirs(yoloindir / 'labels/train', exist_ok = True)
    os.makedirs(yoloindir / 'labels/val', exist_ok = True)
    os.makedirs(yoloindir / 'images/train', exist_ok = True)
    os.makedirs(yoloindir / 'images/val', exist_ok = True)
    inputdir = Path('/home/sofia/Documents/VinBigData/Data')
    labeldir = inputdir / 'yolov5_labels'
    for file in tqdm(train_files):
        shutil.copy(file, yoloindir / 'images/train')
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(labeldir, filename+'.txt'), yoloindir / 'labels/train')
        
    for file in tqdm(val_files):
        shutil.copy(file, yoloindir / 'images/val')
        filename = file.split('/')[-1].split('.')[0]
        shutil.copy(os.path.join(labeldir, filename+'.txt'), yoloindir / 'labels/val')

# make_yolo_train_files(dir=yolodir, fold=fold, single_cls=False, cls=2)

def make_yolo_val300(dir=yolodir, dim=dim, fold=fold):
    val300_files = []
    with open(join(dir, 'yolo_files/val.txt'), 'r') as f:
        for file_path in f.readlines():
            val300_files.append(file_path)
            
    randomlist = random.sample(range(0, 3000), 300)
    val300_files = list(np.array(val300_files)[randomlist])

    with open(join(dir, 'yolo_files/val300.txt'), 'w') as f:
        for path in val300_files:
            f.write(path)
            
    data = dict(
        train =  join(dir , 'yolo_files/train.txt') ,
        val   =  join(dir , 'yolo_files/val300.txt' ),
        nc    = len(classes),
        names = classes
        )
    
    with open(join(dir , 'yolo_files/vinbigdata300.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
    annotations_coco_format(fold=fold, dim=dim, val300=True)
        
# make_yolo_val300(yolodir, dim=dim, fold=fold)

def make_yolo_labels(inputdir=inputdir, yolodir=yolodir, single_cls=True, cls=2):
    labeldir = Path(inputdir / 'yolov5_labels')
    os.makedirs(str(labeldir), exist_ok=True)
    if single_cls:
        train_df = pd.read_csv(yolodir / f'df_norm_folds_{classes[cls]}.csv')
    else:
        train_df = pd.read_csv(yolodir / 'df_norm_folds.csv')
    image_ids = train_df['image_id'].unique()

    label_dict = []
    for i in image_ids:
        i_df = train_df[train_df['image_id'] == i]
        label_list = []
        if i_df['class_id'].unique()[0] == 14:
            label = [14, 0.5, 0.5, 1.0, 1.0]
            label_list.append(label)
        else:
            for index, row in i_df.iterrows():
                label = [row['class_id'], row['x_mid'], row['y_mid'], row['w'], row['h']]
                label_list.append(label)
        label_dict.append({
            'image_id': i, 
            'labels': label_list
        })
    
    for item in label_dict:
        image_id = item['image_id']
        labels = item['labels']
        with open(labeldir/ f'{image_id}.txt', 'w') as f:
            for label in labels:
                for l in label:
                    f.write("%s" % l + ' ')
                f.write("\n")

# make_yolo_labels(inputdir, yolodir, single_cls=False, cls=2)


    
    
    
    
    
    
    