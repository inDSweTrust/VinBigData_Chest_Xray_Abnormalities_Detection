# VinBigData Chest Xray Abnormalities Detection
## Summary
Kaggle [VinBigData Chest Xray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) competition object Detection-Classification pipeline.

The aim was to detect and classify 14 abnormalities in chest radiographs with a YOLOv5 - based pipeline using transfer learning.  

For all classes combined, this model achieves mean Average Precision (mAP) **0.236** at IoU > 0.4.


 <img src="https://user-images.githubusercontent.com/68122114/199068590-ba08e593-97bb-4130-94a8-6eefd42c3bc2.png" width="499" height="570"> <img src="https://user-images.githubusercontent.com/68122114/199068602-9b9e6855-95f5-44c1-bab0-6d679845aabf.png" width="499" height="570">
 

## Sources:
* Ultralytics YOLOv5 - https://github.com/ultralytics/yolov5
* Weighted Box Fusion by @ZFTurbo - https://github.com/ZFTurbo/Weighted-Boxes-Fusion
* Mean Average Precision by @ZFTurbo - https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes
