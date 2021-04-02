import pandas as pd
import numpy as np 
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

def ensemble_models(*pred_df, path, evaluation=False):
    models = []
    for i, pred in enumerate(pred_df):
        if i == 0:
            pred_df0 = pred
        else:
            models.append(pred)
    
    pred_df = pred_df0.append(models)
    image_ids = pred_df["image_id"].unique()
    
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
    
    for i in range(15):
        pred_df.loc[pred_df.class_name == class_dict[i], 'class_name'] = i
    
    # boxes_list = []
    # scores_list = []
    # labels_list = []
    results = []
    for image_id in tqdm(image_ids, total=len(image_ids)):
        # All annotations for the current image.
        data = pred_df[pred_df["image_id"] == image_id]
        data = data.reset_index(drop=True)
        
        box_list = []
        score_list = []
        label_list = []
        # Loop through all of the annotations
        for idx, row in data.iterrows():
            box_list.append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])
            score_list.append(row["score"])
            label_list.append(row["class_name"])
            
        boxes_list = [box_list]
        scores_list = [score_list]
        labels_list = [label_list]
    
        # Calculate WBF
        iou_thr=0.6
        skip_box_thr=0.0001
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=None,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
    
        # Create df for evaluation, format for mAP by @ZFTurbo

        for idx, box in enumerate(boxes):
            if evaluation:
                results.append({
                    "image_id": image_id,
                    "class_name": class_dict[int(labels[idx])],
                    "score": scores[idx],
                    "x_min": box[0],
                    "x_max": box[2],
                    "y_min": box[1],
                    "y_max": box[3]
                })
            else:
                results.append({
                    "image_id": image_id,
                    "class_name": int(labels[idx]),
                    "score": scores[idx],
                    "x_min": box[0],
                    "y_min": box[1],
                    "x_max": box[2],
                    "y_max": box[3]
                })
                
    results = pd.DataFrame(results)
    results.to_csv(path / 'pred_df_ensembled.csv', index=False)

