import numpy as np
import json

def getPositiveCounts():
    with open('labels/train_label_cleaned.json', 'r') as file:
        trainData = json.load(file)
        
    all_labels = [lbl 
              for clip in trainData 
              for lbl in clip['labels']]
    counts = np.bincount(all_labels, minlength=17)
    
    return counts
    
print(getPositiveCounts())