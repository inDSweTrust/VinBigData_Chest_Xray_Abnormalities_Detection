import os
import pandas as pd

from map_boxes import mean_average_precision_for_boxes
from pathlib import Path

from ensemble import ensemble_models
from config import FlagsDet, flags_yolo

# Config
flags = FlagsDet().update(flags_yolo)
dim = flags.dim
fold = flags.fold
filtered = False
ensemble = True
evaldir = Path(flags.evaldir)
os.makedirs(str(evaldir), exist_ok=True)
yolodir = Path(flags.yolodir)


def generate_anno(fold=2, indir=yolodir, outdir=evaldir):
    annotations_df = pd.read_csv(indir / 'df_norm_folds.csv')
    annotations_df = annotations_df[annotations_df.fold == fold]
    annotations_df = annotations_df.drop(columns=['class_id', 'rad_id', 'width', 'height', 
                                                  'image_path', 'x_mid', 'y_mid', 'w', 'h', 
                                                  'area', 'fold'])
    # Write anno to CSV
    annotations_df.reset_index(drop=True, inplace=True)
    annotations_df = annotations_df[['image_id','class_name','x_min','x_max','y_min','y_max']]
    annotations_df.to_csv(outdir / 'annotations_df.csv', index=False)
    

def generate_pred(fold=2, dim=512, filtered=True, single_cls=True, cls=2, indir=yolodir, outdir=evaldir):
    class_dict = {
        0: 'Aortic enlargement',
        1: 'Atelectasis',
        2: 'Calcification',
        3: 'Cardiomegaly',
        4: 'Consolidation',
        5: 'ILD',
        6: 'Infiltration',
        7: 'Lung Opacity',
        8: 'Nodule/Mass',
        9: 'Other lesion',
        10: 'Pleural effusion',
        11: 'Pleural thickening',
        12: 'Pneumothorax',
        13: 'Pulmonary fibrosis',
        14: "No finding"
    }
    # best pred from YOLO
    pred_df = pd.read_json(indir / 'yolov5/runs/test/exp/best_predictions.json')

    # if single_cls:
    #     pred_df.loc[pred_df.category_id==0, 'category_id'] = cls
    pred_df.insert(1, "class_name",0)
    pred_df.insert(5, "x_min",0)
    pred_df.insert(6, "y_min",0)
    pred_df.insert(7, "x_max",0)
    pred_df.insert(8, "y_max",0)
    pred_df.insert(9, "w",0)
    pred_df.insert(10, "h",0)
    
    bbox_vals = pred_df['bbox'].apply(lambda x: pd.Series([i for i in x]))
    pred_df[['x_min', 'y_min', 'w', 'h']] = bbox_vals
    pred_df['x_max'] = pred_df['x_min'] + pred_df['w']
    pred_df['y_max'] = pred_df['y_min'] + pred_df['h']
    

    pred_df = pred_df.drop(columns=['bbox', 'w', 'h'])
    pred_df.x_min = pred_df['x_min'].div(dim)
    pred_df.y_min = pred_df['y_min'].div(dim)
    pred_df.x_max = pred_df['x_max'].div(dim)
    pred_df.y_max = pred_df['y_max'].div(dim)
    
  
    for i in range(15):
        pred_df.loc[pred_df.category_id == i, 'class_name'] = class_dict[i]

    pred_df = pred_df.drop(columns=['category_id'])
    pred_df = pred_df[['image_id','class_name','score','x_min','x_max','y_min','y_max']]
    
    # Clean predictions
    if filtered:
        # Find No finding images with conf threshold and del all other preds 
        conf_thresh = 0.89
        # conf_thresh = 0.976
        assert pred_df.loc[pred_df.score >= conf_thresh, 'class_name'].unique()[0] == 'No finding'
        image_ids = pred_df.loc[pred_df.score >= conf_thresh, 'image_id'].unique()
        for i in image_ids:
            index = pred_df[pred_df.image_id == i].index
            pred_df = pred_df.drop(index[1:])
        pred_df.reset_index(drop=True, inplace=True)
        
        # Del No finding duplicates for all image ids
        for i in pred_df[pred_df.class_name=='No finding']['image_id'].unique():
            index = pred_df[(pred_df.image_id == i) & (pred_df.class_name=='No finding')].index
            pred_df = pred_df.drop(index[1:])
        pred_df.reset_index(drop=True, inplace=True)
    
    # Filter out extra single class preds
    # class_thresh = 0.8
    # if single_cls:
    #     pred_df = pred_df[pred_df.score > class_thresh]
    #     pred_df.reset_index(drop=True, inplace=True)
    
    if filtered:
        pred_df.to_csv(outdir / 'pred_df_filtered.csv', index=False)
    elif single_cls:
        pred_df.to_csv(outdir / f'pred_df_{class_dict[cls]}.csv', index=False)
    else:
        pred_df.to_csv(outdir / 'pred_df.csv', index=False)
        
def evaluate(*pred_df, outdir=evaldir, filtered=False, ensemble=False):
    ann = pd.read_csv(outdir / 'annotations_df.csv')
    if filtered:
        det = pd.read_csv(outdir / 'pred_df_filtered.csv')
    elif ensemble:
        preds = []
        for i, pred in enumerate(pred_df):
            preds.append(pred)
        ensemble_models(preds[0], preds[1], path=evaldir, evaluation=True)
        det = pd.read_csv(outdir / 'pred_df_ensembled.csv')
    else:
        det = pd.read_csv(outdir / 'pred_df.csv')
        
    ann = ann[['image_id', 'class_name', 'x_min', 'x_max', 'y_min', 'y_max']].values
    det = det[['image_id', 'class_name', 'score', 'x_min', 'x_max', 'y_min', 'y_max']].values
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.4)
    
    
# generate_anno(fold=fold)
# generate_pred(fold=fold, dim=dim, filtered=filtered, single_cls=False, cls=2)
# pred_df1 = pd.read_csv(evaldir / 'pred_df1.csv')
# pred_df2 = pd.read_csv(evaldir / 'pred_df2.csv')
# evaluate(pred_df1, pred_df2, outdir=evaldir, filtered=False, ensemble=True)
evaluate(outdir=evaldir, filtered=False, ensemble=False)
