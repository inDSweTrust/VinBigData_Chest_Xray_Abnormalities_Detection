from pathlib import Path

evaldir = Path('/home/sofia/Documents/VinBigData/Results/YOLOv5/Train_ALL/00_Eval')

def read_eval_results(evaldir, file='IMG512_WBF_mAP'):
    f = open(evaldir / file, 'r')
    lines = f.readlines()
    
    results = {}
    for cls in range(len(lines)):
        string = lines[cls].split()
        if len(string) == 6:
            results[f"{string[0]} {string[1]}"] = float(string[3])
        elif len(string) == 5:
            results[f"{string[0]}"] = float(string[2])
        elif len(string) == 2:
            results[f"{string[0]}"] = float(string[1])
    return results

def compare_results(class_names, model1, model2, compare=''):
    print(f'{compare}')
    print('********************')
    for cls in class_names:
        if model1[cls] < model2[cls]:
            print(f'{cls} - BL: {model1[cls]}, AFTER: {model2[cls]}')
    
    
    
IMG512 = read_eval_results(evaldir, 'IMG512_WBF_mAP')
IMG1024 = read_eval_results(evaldir, 'IMG1024_WBF_mAP')
IMG512_CLAHE = read_eval_results(evaldir, 'IMG512_WBF_CLAHE_mAP')
IMG1024_CLAHE = read_eval_results(evaldir, 'IMG1024_WBF_CLAHE_mAP')
IMG512_Hist = read_eval_results(evaldir, 'IMG512_WBF_Hist_mAP')
# IMG1024_Hist = read_eval_results(evaldir, 'IMG1024_WBF_Hist_mAP')


class_names = [k for k in IMG512]
compare_results(class_names, IMG512, IMG1024, compare='Image Size Comparison')
compare_results(class_names, IMG512, IMG512_CLAHE, compare='IMG512 to IMG512_CLAHE Comparison')
compare_results(class_names, IMG512, IMG1024_CLAHE, compare='IMG512 to IMG1024_CLAHE Comparison')
compare_results(class_names, IMG512, IMG512_Hist, compare='IMG512 to IMG512_Hist Comparison')
compare_results(class_names, IMG512_CLAHE, IMG512_Hist, compare='IMG512_CLAHE to IMG512_Hist Comparison')


# IMG512_CLAHE
# IMG512
