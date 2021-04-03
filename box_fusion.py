import pandas as pd

from pathlib import Path
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
from config import FlagsDet, flags_yolo

# Config
flags = FlagsDet().update(flags_yolo)
inputdir = Path(flags.inputdir)

#Based on  https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/208468
def wbf(df, iou_thr, skip_box_thr, inputdir):
    df.fillna(0, inplace=True)
    df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0
    
    results = []
    image_ids = df["image_id"].unique()
    
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
    
    for image_id in tqdm(image_ids, total=len(image_ids)):
        # All annotations for the current image.
        data = df[df["image_id"] == image_id]
        data = data.reset_index(drop=True)
        
        annotations = {}
        weights = []
    
        # WBF expects the coordinates in 0-1 range.
        max_value = data.iloc[:, 4:8].values.max()
        data.loc[:, ["x_min", "y_min", "x_max", "y_max"]] = data.iloc[:, 4:8] / max_value
    
        # Loop through all of the annotations
        for idx, row in data.iterrows():
    
            rad_id = row["rad_id"]

            if rad_id not in annotations:
                annotations[rad_id] = {
                    "boxes_list": [],
                    "scores_list": [],
                    "labels_list": [],
                }
    
                # We consider all of the radiologists as equal.
                weights.append(1.0)
    
            annotations[rad_id]["boxes_list"].append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])
            annotations[rad_id]["scores_list"].append(1.0)
            annotations[rad_id]["labels_list"].append(row["class_id"])

        boxes_list = []
        scores_list = []
        labels_list = []
    
        for annotator in annotations.keys():
            boxes_list.append(annotations[annotator]["boxes_list"])
            scores_list.append(annotations[annotator]["scores_list"])
            labels_list.append(annotations[annotator]["labels_list"])
            
            boxes_list
            labels_list
        # Calculate WBF
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
        
        for idx, box in enumerate(boxes):
            results.append({
                "image_id": image_id,
                "class_name": class_dict[int(labels[idx])],
                "class_id": int(labels[idx]),
                "rad_id": "wbf",
                "x_min": box[0] * max_value,
                "y_min": box[1] * max_value,
                "x_max": box[2] * max_value,
                "y_max": box[3] * max_value,
                "width": data['width'].unique()[0],
                "height": data['height'].unique()[0]
            })
    
    results = pd.DataFrame(results)
    results.to_csv(inputdir / 'train_wh_wbf.csv', index=False)
    return results

train_df = pd.read_csv(inputdir / 'train_wh.csv')
wbf(df=train_df, iou_thr=0.6, skip_box_thr=0.0001, inputdir=inputdir)

