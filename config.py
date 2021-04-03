from dataclasses import dataclass, field
from typing import Dict


flags_yolo = {   
    "inputdir": "/home/sofia/Documents/VinBigData/Data",
    "yolodir": "/home/sofia/Documents/VinBigData/VinBigData_YOLOv5", 
    "outdir": "/home/sofia/Documents/VinBigData/VinBigData_YOLOv5/output",
    "evaldir": "/home/sofia/Documents/VinBigData/VinBigData_YOLOv5/eval",
    "dim": 512,
    "fold": 2,
    "use_class14": True
}

@dataclass
class FlagsDet():
    # General
    inputdir: str = "/home/sofia/Documents/VinBigData/Data"
    yolodir: str = "/home/sofia/Documents/VinBigData/VinBigData_YOLOv5"
    outdir: str = "/home/sofia/Documents/VinBigData/VinBigData_YOLOv5/output"
    evaldir: str = "/home/sofia/Documents/VinBigData/VinBigData_YOLOv5/eval"
    dim: int = 512
    fold: int = 2
    use_class14: bool =  True
      
    def update(self, param_dict: Dict) -> "FlagsDet":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self