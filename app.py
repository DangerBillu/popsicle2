import os
import requests
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_file, render_template
from dotenv import load_dotenv
from PIL import Image
from flask_cors import CORS
import cv2
import numpy as np
from skimage.exposure import match_histograms

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'

BG_REMOVAL_API = "https://hf.space/embed/not-lain/background-removal/api/predict/"
TEXT_TO_IMAGE_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"
}

# In-memory storage for layers and assets (for demonstration)
layers = []
assets = []

def query_huggingface_api(api_url, **kwargs):
    local_headers = HEADERS.copy()
    if "json" in kwargs:
        local_headers["Content-Type"] = "application/json"
    response = requests.post(api_url, headers=local_headers, **kwargs)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    return response.content

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            image_bytes = file.read()
        elif request.is_json and 'image' in request.json:
            image_bytes = base64.b64decode(request.json['image'])
        else:
            return jsonify({"error": "No image provided"}), 400

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"data": [image_b64]}
        
        processed_image = query_huggingface_api(BG_REMOVAL_API, json=payload)

        layer_id = len(layers) + 1
        layers.append({
            "id": layer_id,
            "image": processed_image,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        })
        assets.append(processed_image)

        return jsonify({"image": base64.b64encode(processed_image).decode("utf-8")})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        payload = {"inputs": data['prompt']}
        generated_image = query_huggingface_api(TEXT_TO_IMAGE_API, json=payload)
        
        img = Image.open(BytesIO(generated_image))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        processed_image = img_byte_arr.getvalue()

        layer_id = len(layers) + 1
        layers.append({
            "id": layer_id,
            "image": processed_image,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        })
        assets.append(processed_image)

        return jsonify({"image": base64.b64encode(processed_image).decode('utf-8')})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/apply-mockup-advanced', methods=['POST'])
def apply_mockup_advanced():
    if 'base' not in request.files or 'design' not in request.files:
        return jsonify({"error": "Both base and design images are required."}), 400

    base_file = request.files['base']
    design_file = request.files['design']
    base_bytes = base_file.read()
    design_bytes = design_file.read()

    base_img = cv2.imdecode(np.frombuffer(base_bytes, np.uint8), cv2.IMREAD_COLOR)
    design_img = cv2.imdecode(np.frombuffer(design_bytes, np.uint8), cv2.IMREAD_COLOR)
    if base_img is None or design_img is None:
        return jsonify({"error": "Could not decode one or both images."}), 400

    # Perspective Correction – using fixed demo destination points.
    pts_dst = np.array([[100, 200], [400, 180], [420, 380], [120, 400]], dtype=np.float32)
    h_design, w_design = design_img.shape[:2]
    pts_src = np.array([[0, 0], [w_design, 0], [w_design, h_design], [0, h_design]], dtype=np.float32)
    M, status = cv2.findHomography(pts_src, pts_dst)
    warped_design = cv2.warpPerspective(design_img, M, (base_img.shape[1], base_img.shape[0]))

    # Smart Masking – create a binary mask for the destination quadrilateral.
    mask = np.zeros((base_img.shape[0], base_img.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)

    # Histogram Matching – use channel_axis=-1 instead of multichannel=True.
    warped_design_float = warped_design.astype(np.float32)
    base_img_float = base_img.astype(np.float32)
    matched_design = match_histograms(warped_design_float, base_img_float, channel_axis=-1)
    matched_design = np.clip(matched_design, 0, 255).astype(np.uint8)

    # Composite the matched design over the base using the mask.
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    base_float = base_img.astype(np.float32)
    design_float = matched_design.astype(np.float32)
    composite = (design_float * mask_3ch + base_float * (1 - mask_3ch)).astype(np.uint8)

    retval, buffer = cv2.imencode('.png', composite)
    return send_file(BytesIO(buffer.tobytes()), mimetype='image/png', download_name='advanced_mockup.png')

@app.route('/layers', methods=['GET'])
def get_layers():
    return jsonify([{
        "id": layer['id'],
        "position": layer['position'],
        "scale": layer['scale']
    } for layer in layers])

@app.route('/assets', methods=['GET'])
def get_assets():
    return jsonify([{
        "id": idx,
        "preview": f"/asset/{idx}"
    } for idx in range(len(assets))])

@app.route('/asset/<int:asset_id>', methods=['GET'])
def get_asset(asset_id):
    if asset_id >= len(assets):
        return jsonify({"error": "Asset not found"}), 404
    return send_file(BytesIO(assets[asset_id]), mimetype='image/png')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
