import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from video_transformer import VideoTransformer

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
CORS(app)  # enable CORS for all domains (configure in prod)

def predict(video_path: str):
    frames = getFramesFromVideo(video_path)
    model = VideoTransformer()
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

    # limit to first 60 frames and add batch dim
    frames = frames[:60]
    return torch.stack(frames, dim=0).unsqueeze(0)


@app.route('/api/predict', methods=['POST'])
def getPrediction():
    """
    Flask endpoint to handle prediction requests.
    Expects JSON: {"videoPath": "path/to/video.mp4"}
    Returns: {"events": [{"action": str, "confidence": float}, ...]}
    """
    data = request.get_json()
    video_path = data.get('videoPath')
    if not video_path:
        return jsonify({'error': 'videoPath is required'}), 400

    try:
        result = predict(video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)