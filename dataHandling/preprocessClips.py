# Imports
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import torch
from torchvision import transforms
import os
import json
import math
import cv2
from glob import glob
import multiprocessing

# Data Processing

# Configuration
DATA_DIR = 'SoccerNet/data'
OUTPUT_DIR = 'SoccerNet/processedData'
FPS = 3
CLIP_DURATION = 60  # seconds

def process_clip(args):
    """
    Process a single clip: extract frames and write labels.json
    args = (clip_id, split_name, match_path, half, clip_idx)
    """
    clip_id, split_name, match_path, half, clip_idx = args

    # Load labels
    labels_file = os.path.join(match_path, 'Labels-v2.json')
    with open(labels_file, 'r') as f:
        match_data = json.load(f)
    annotations = match_data.get('annotations', [])

    # Build list of (absolute_sec, ann) for this half
    half_anns = []
    for ann in annotations:
        h, t = ann['gameTime'].split(' - ')
        if int(h) != half:
            continue
        mm, ss = map(int, t.split(':'))
        secs = mm * 60 + ss
        half_anns.append((secs, ann))

    # Open video
    video_file = os.path.join(match_path, f'{half}_224p.mkv')
    cap = cv2.VideoCapture(video_file)

    # Compute clip start/end
    start = clip_idx * CLIP_DURATION
    end = start + CLIP_DURATION

    # Prepare output folder
    out_folder = os.path.join(OUTPUT_DIR, split_name, str(clip_id))
    os.makedirs(out_folder, exist_ok=True)

    # Extract frames at FPS rate
    for f_idx in range(FPS * CLIP_DURATION):
        t_sec = start + f_idx / FPS
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_folder, f"{f_idx+1}.jpg"), frame)

    cap.release()

    # Build clip‐relative annotations
    clip_anns = []
    for secs, ann in half_anns:
        if start <= secs < end:
            rel = secs - start
            rel_mm = int(rel // 60)
            rel_ss = int(rel % 60)
            new_gt = f"{half} - {rel_mm:02d}:{rel_ss:02d}"
            new_pos = int(rel * FPS)
            new_ann = ann.copy()
            new_ann['gameTime'] = new_gt
            new_ann['position'] = str(new_pos)
            clip_anns.append(new_ann)

    # Write labels.json
    out_labels = {
        'UrlLocal':      match_data.get('UrlLocal', ''),
        'UrlYoutube':    match_data.get('UrlYoutube', ''),
        'annotations':   clip_anns,
        'gameAwayTeam':  match_data.get('gameAwayTeam', ''),
        'gameDate':      match_data.get('gameDate', ''),
        'gameHomeTeam':  match_data.get('gameHomeTeam', ''),
        'gameScore':     match_data.get('gameScore', '')
    }
    with open(os.path.join(out_folder, 'labels.json'), 'w') as f:
        json.dump(out_labels, f, indent=2)


# Gather all match folders
match_paths = sorted(glob(os.path.join(DATA_DIR, '*', '*', '*')))
num_matches = len(match_paths)

# Split indices
train_end = int(num_matches * 0.7)
valid_end = train_end + int(num_matches * 0.2)

splits = {
    'train': match_paths[:train_end],
    'valid': match_paths[train_end:valid_end],
    'test':  match_paths[valid_end:]
}

# Build list of all clips to process, assigning each a unique clip_id
tasks = []
clip_id = 0
for split_name, paths in splits.items():
    for match_path in paths:
        for half in [1, 2]:
            video_file = os.path.join(match_path, f'{half}_224p.mkv')
            if not os.path.isfile(video_file):
                continue

            cap = cv2.VideoCapture(video_file)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            duration = total_frames / orig_fps
            num_clips = int(duration // CLIP_DURATION)
            cap.release()

            for i in range(num_clips):
                tasks.append((clip_id, split_name, match_path, half, i))
                clip_id += 1

# Process all clips in parallel (one task per core by default)
with multiprocessing.Pool() as pool:
    pool.map(process_clip, tasks)