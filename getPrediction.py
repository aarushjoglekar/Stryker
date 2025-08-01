import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
import tempfile
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_cors import CORS
from video_transformer import VideoTransformer
import traceback
import ssl
from collections import OrderedDict

ssl._create_default_https_context = ssl._create_unverified_context

labels2ids = {
    "Ball out of play": 0,
    "Clearance": 1,
    "Corner": 2,
    "Direct free-kick": 3,
    "Foul": 4,
    "Goal": 5,
    "Indirect free-kick": 6,
    "Kick-off": 7,
    "Offside": 8,
    "Penalty": 9,
    "Red card": 10,
    "Shots off target": 11,
    "Shots on target": 12,
    "Substitution": 13,
    "Throw-in": 14,
    "Yellow card": 15,
    "Yellow->red card": 16
}

labelsStrings = list(labels2ids.keys())  # corrected to list

app = Flask(__name__)
# CORS(app)  # enable CORS for all domains (configure in prod)
CORS(app, resources={ r"/api/*": { "origins": "http://localhost:8080" } })


def predict(video_path: str):
    frames = getFramesFromVideo(video_path)
    model = VideoTransformer(num_frames_per_clip = 60)
    
    raw_ckpt = torch.load('models/best_model.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in raw_ckpt.items():
        # remove the "module." prefix
        new_key = k.replace('module.', '')  
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    output = model.forward(frames)
    probs = torch.sigmoid(output)[0]  # assume batch dim
    events = []
    for idx, probability in enumerate(probs):
        if probability.item() > 0.5:
            events.append({
                'action': labelsStrings[idx],
                'confidence': probability.item()
            })
    return {'events': events}


def getFramesFromVideo(video_path: str):
    out_dir = "frames"
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Couldn't read FPS from video!")

    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    frames = []
    t = 0.0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        success, frame = cap.read()
        if not success:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor_img = to_tensor(img)
        frames.append(tensor_img)
        t += 1.0

    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted from video.")

    frames = frames[:60]
    return torch.stack(frames, dim=0).unsqueeze(0)

@app.route('/api/predict', methods=['POST'])
def getPrediction():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    file.save(temp_path)

    try:
        result = predict(temp_path)
        return jsonify(result)
    except Exception as e:
        # <-- print the full traceback so you know exactly what went wrong
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except OSError:
            pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)