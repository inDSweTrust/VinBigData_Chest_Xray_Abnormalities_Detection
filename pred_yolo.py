import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from ensemble import ensemble_models
from pathlib import Path
from config import FlagsDet, flags_yolo

# Config
flags = FlagsDet().update(flags_yolo)
dim = flags.dim
inputdir = Path(flags.inputdir)
outdir = Path(flags.outdir)
test_df = pd.read_csv(inputdir / f'images_{dim}/test_meta_yolo.csv')

# Based on https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-infer
def yolo2voc(image_height, image_width, bboxes, ensemble=False):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    # Normalized boxes are required for ensemble
    if ensemble:        
        bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
        bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    else:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
        
        bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
        bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes

def delete_class(pred_df):
    pred_df = pred_df.drop(index=pred_df[pred_df.class_name==14].index)

def generate_det_df(test_df, ensemble=False, outdir = outdir, name=''):
    """
    Generates a predictions dataframe from YOLOv5 det files which can be 
    directly used to create submission.csv or used for ensembling.
    
    """ 
    results = []
    
    for file_path in tqdm(glob(flags.yolodir + '/yolov5/runs/detect/exp/labels/*txt')):
        image_id = file_path.split('/')[-1].split('.')[0]
        w, h = test_df.loc[test_df.image_id==image_id,['width', 'height']].values[0]
        f = open(file_path, 'r')
        data = np.array(f.read().replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6)

        data = data[:, [0, 5, 1, 2, 3, 4]]
        data = data[::-1]

        class_conf = data[:, :2]
        if ensemble:
            bbox = np.round(yolo2voc(h, w, data[:, 2:], ensemble=True),4)
        else:
            bbox = np.round(yolo2voc(h, w, data[:, 2:], ensemble=False))
        # Create df with test predictions - use for single pred or ensemble
        for idx, box in enumerate(bbox):
            results.append({
                "image_id": image_id,
                "class_name": class_conf[idx][0],
                "score": class_conf[idx][1],
                "x_min": box[0],
                "y_min": box[1],
                "x_max": box[2],
                "y_max": box[3]
            })
        
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(outdir / f'pred_{name}.csv',index = False)
    
    
def generate_submission(pred_df, test_df, outdir=outdir, ensemble=True):
    """
    Generates submission.csv from pred csv file (single model or ensemble)
    
    """ 
    image_ids = pred_df.image_id.unique()

    image_ids_list = []
    predictions = []
    with tqdm(total=len(image_ids)) as pbar:
        for i in image_ids:
            # i = '024e693014e70d448f650267bed661ae'
            w, h = test_df.loc[test_df.image_id==i,['width', 'height']].values[0]
            image_df = pred_df[pred_df.image_id==i].astype({"class_name": int,
                                                            "x_min": float,
                                                            "y_min": float,
                                                            "x_max": float,
                                                            "y_max": float
                                                            }).round({"score": 4})
            if ensemble:
                image_df['x_min'] = image_df['x_min'].apply(lambda x: x * w).astype('int')
                image_df['y_min'] = image_df['y_min'].apply(lambda x: x * h).astype('int')
                image_df['x_max'] = image_df['x_max'].apply(lambda x: x * w).astype('int')
                image_df['y_max'] = image_df['y_max'].apply(lambda x: x * h).astype('int')
            
            # Replace "No finding" with [1.0, 0, 0, 1, 1] and drop extra 'No finding' predictions
            no_finding_index = image_df[image_df.class_name==14][:1].index
            image_df.loc[no_finding_index, ['x_min', 'y_min', 'x_max', 'y_max']] = [0, 0, 1, 1]
            image_df = image_df.drop(index=image_df[image_df.class_name==14][1:].index)

            class_name = image_df.class_name.values.reshape(-1, 1)
            score = image_df.score.values.reshape(-1, 1)
            bbox = image_df.iloc[:, 3:].to_numpy()

            # Filter Normal preds
            if class_name[0]==14 and score[0] >= 0.976:
                pred_str = ' '.join(['14', '1.0', '0', '0', '1', '1'])
            else:                  
                pred_str = list(np.round(np.concatenate((class_name, score, bbox), axis =1).reshape(-1), 1).astype(str))
                for idx in range(len(pred_str)):
                    pred_str[idx] = str(int(float(pred_str[idx]))) if idx%6!=1 else pred_str[idx]
                pred_str = ' '.join(pred_str)
            
            image_ids_list.append(i)    
            predictions.append(pred_str)
            pbar.update(1)

    pred_df = pd.DataFrame({'image_id':image_ids_list,
                            'PredictionString': predictions})
    submission_df = pd.merge(test_df, pred_df, on = 'image_id', how = 'left')
    submission_df = submission_df[['image_id', 'PredictionString']]
    submission_df.to_csv(outdir / 'submission.csv', index = False)
    
# generate_det_df(test_df, ensemble=True, outdir=outdir, name='norm')

# pred_df1 = pd.read_csv(outdir / 'pred_512.csv')
# pred_df2 = pd.read_csv(outdir / 'pred_1024.csv')
# pred_df3 = pd.read_csv(outdir / 'pred_norm.csv')
# ensemble_models(pred_df1, pred_df2, pred_df3, path=outdir, evaluation=False)

generate_submission(pred_df=pd.read_csv(outdir / 'pred_norm.csv'), test_df=test_df, ensemble=False)
# generate_submission(pred_df=pd.read_csv(outdir / 'pred_df_ensembled.csv'), test_df=test_df, outdir=outdir, ensemble=True)
