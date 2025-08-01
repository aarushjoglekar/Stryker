import os
import json

def preprocess_labels():
    BASE_DIR = "SoccerNet/processedData"
    SPLITS   = ("train", "valid", "test")

    label2id = {'Ball out of play': 0, 'Clearance': 1, 'Corner': 2, 'Direct free-kick': 3, 'Foul': 4, 'Goal': 5, 'Indirect free-kick': 6, 'Kick-off': 7, 'Offside': 8, 'Penalty': 9, 'Red card': 10, 'Shots off target': 11, 'Shots on target': 12, 'Substitution': 13, 'Throw-in': 14, 'Yellow card': 15, 'Yellow->red card': 16}

    # Write out your mapping so you can inspect or reuse it
    with open("SoccerNet/label2id.json", "w") as f:
        json.dump(label2id, f, indent=4)
    print(f"Found {len(label2id)} classes, wrote label2id.json")

    # Build train/valid/test label files
    for split in SPLITS:
        out = []
        split_dir = os.path.join(BASE_DIR, split)

        for clip in sorted(os.listdir(split_dir), key=int):
            labels_path = os.path.join(split_dir, clip, "labels.json")
            image_dir = os.path.join(split_dir, clip)  # Example image path, not used here

            data = json.load(open(labels_path, "r"))
            # ids  = {
            #     label2id[ann["label"]]
            #     for ann in data.get("annotations", [])
            #     if ann["label"] in label2id
            # }
            # out[clip] = sorted(ids)
            anno = data['annotations']
            labels = [label2id[ann["label"]] for ann in anno]
            labels_str = [ann["label"] for ann in anno]
            out.append({
                "image_dir": image_dir,
                "labels": labels,
                "labels_str": labels_str,
            })

        fname = f"SoccerNet/{split}_label.json"
        with open(fname, "w") as f:
            json.dump(out, f, indent=4)

        print(f"Wrote {len(out)} entries to {fname}")

def remove_no_labels():
    """
    Remove entries with no labels from the dataset.
    """
    for split in ["train", "valid", "test"]:
        fname = f"SoccerNet/{split}_label.json"
        data = json.load(open(fname, "r"))
        filtered_data = [entry for entry in data if len(entry["labels"]) > 0]

        with open(fname, "w") as f:
            json.dump(filtered_data, f, indent=4)

        print(f"Removed {len(data) - len(filtered_data)} entries with no labels from {fname}")

def remove_no_images():
    """
    Remove entries with no images from the dataset. (Removed 9 samples in training set)
    """
    for split in ["train", "valid", "test"]:
        db = f"SoccerNet/{split}_label.json"
        data = json.load(open(db, "r"))
        
        for entry in data:
            image_dir = entry["image_dir"]
            
            for i in range(1, 180 + 1):
                frame_path = f"{image_dir}/{i}.jpg"
                if not os.path.exists(frame_path):
                    print(f"Removing {image_dir} as it has no image {i}.jpg")
                    data.remove(entry)
                    break
        
        new_file = f"SoccerNet/{split}_label_cleaned.json"
        with open(new_file, "w") as f:
            json.dump(data, f, indent=4)
        

# preprocess_labels()
# remove_no_labels()
remove_no_images()