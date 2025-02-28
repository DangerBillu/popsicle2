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
import json
import torch
from transformers import AutoModelForDepthEstimation
import torchvision.transforms as T

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize models and API endpoints
model = None
midas_transforms = None
BG_REMOVAL_API = "https://hf.space/embed/not-lain/background-removal/api/predict/"
TEXT_TO_IMAGE_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN', '')}"}

# File-based storage settings
LAYERS_FILE = 'data/layers.json'
ASSETS_FOLDER = 'data/assets'
layers = []
assets = []

def ensure_numpy(img):
    """
    Convert input image to a numpy array if it is a PIL Image or similar.
    """
    if isinstance(img, Image.Image):
        return np.array(img)
    return np.asarray(img)

def load_models():
    global model, midas_transforms
    try:
        # Load the official MiDaS model and transforms from torch.hub
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        model.eval()
        app.logger.info("MiDaS model and transforms loaded successfully")
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        raise

def estimate_depth(pil_img):
    try:
        # Convert PIL image to RGB and apply MiDaS transforms
        img = pil_img.convert("RGB")
        transformed = midas_transforms(img).unsqueeze(0)
        with torch.no_grad():
            prediction = model(transformed)
            # Interpolate prediction to original image size (PIL size is (width, height))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
        depth_map = prediction.squeeze().cpu().numpy()
        # Normalize depth map to the [0, 1] range
        return cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    except Exception as e:
        app.logger.error(f"Depth estimation failed: {str(e)}")
        raise

def ensure_data_dirs():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(ASSETS_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(LAYERS_FILE), exist_ok=True)
    global layers
    if os.path.exists(LAYERS_FILE):
        try:
            with open(LAYERS_FILE, 'r') as f:
                layers = json.load(f)
        except json.JSONDecodeError:
            layers = []
    global assets
    asset_files = [f for f in os.listdir(ASSETS_FOLDER) if f.endswith('.png')]
    asset_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'))
    assets.extend([open(os.path.join(ASSETS_FOLDER, f), 'rb').read() for f in asset_files])

def save_layers():
    with open(LAYERS_FILE, 'w') as f:
        json.dump(layers, f)

def save_asset(asset_data):
    asset_id = len(assets)
    assets.append(asset_data)
    with open(os.path.join(ASSETS_FOLDER, f"{asset_id}.png"), 'wb') as f:
        f.write(asset_data)
    return asset_id

def query_huggingface_api(api_url, **kwargs):
    try:
        response = requests.post(api_url, headers=HEADERS, **kwargs, timeout=30)
        if response.status_code == 503:
            return None, "Model is loading, please try again in a moment"
        return response.content, None
    except Exception as e:
        return None, str(e)

def compute_displacement_field(depth_map, displacement_strength=15.0):
    # Ensure depth_map is a NumPy array
    depth_map = ensure_numpy(depth_map)
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=5)
    h, w = depth_map.shape
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32) + grad_x * displacement_strength
    map_y = map_y.astype(np.float32) + grad_y * displacement_strength
    return map_x, map_y

def warp_design_image(design_img, map_x, map_y):
    # Ensure design_img is a NumPy array
    design_img = ensure_numpy(design_img)
    if len(design_img.shape) == 3 and design_img.shape[2] == 3:
        return cv2.remap(design_img, map_x, map_y, cv2.INTER_LINEAR)
    elif len(design_img.shape) == 3 and design_img.shape[2] == 4:
        # Handle RGBA images by processing channels separately
        result = np.zeros_like(design_img)
        for c in range(design_img.shape[2]):
            result[:,:,c] = cv2.remap(design_img[:,:,c], map_x, map_y, cv2.INTER_LINEAR)
        return result
    else:
        return cv2.remap(design_img, map_x, map_y, cv2.INTER_LINEAR)

def apply_surface_texture(design, base, depth_map):
    # Ensure inputs are NumPy arrays
    design = ensure_numpy(design)
    base = ensure_numpy(base)
    depth_map = ensure_numpy(depth_map)
    
    # Convert to float32 for calculations
    design = design.astype(np.float32) / 255.0
    base = base.astype(np.float32) / 255.0
    
    # Compute gradients of the depth map
    zy, zx = np.gradient(depth_map)
    normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm == 0] = 1  # Avoid division by zero
    normal /= norm
    
    # Set up lighting for texture effect
    light = np.array([0.5, 0.5, 1.0])
    light = light / np.linalg.norm(light)  # Normalize light vector
    
    # Calculate lighting intensity based on surface normals
    intensity = np.clip(np.dot(normal.reshape(-1, 3), light).reshape(depth_map.shape), 0, 1)
    
    # Apply lighting to design
    textured = design * intensity[..., np.newaxis]
    
    # Create alpha mask based on depth
    alpha = 0.7 * (1.0 - depth_map[..., np.newaxis])
    
    # Composite the warped design onto the base image
    result = (textured * alpha + base * (1 - alpha)) * 255
    
    return result

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    try:
        if 'file' not in request.files and not (request.is_json and 'image' in request.json):
            return jsonify({"error": "No image provided"}), 400

        image_bytes = (request.files['file'].read() 
                       if 'file' in request.files 
                       else base64.b64decode(request.json['image']))

        payload = {"data": [base64.b64encode(image_bytes).decode("utf-8")]}
        processed_image, error = query_huggingface_api(BG_REMOVAL_API, json=payload)
        if error:
            return jsonify({"error": error}), 500

        asset_id = save_asset(processed_image)
        layer = {"id": len(layers)+1, "asset_id": asset_id, "position": {"x":0, "y":0}, "scale":1.0}
        layers.append(layer)
        save_layers()

        return jsonify({
            "image": base64.b64encode(processed_image).decode("utf-8"),
            "layer_id": layer["id"],
            "asset_id": asset_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        generated_image, error = query_huggingface_api(TEXT_TO_IMAGE_API, json={"inputs": data['prompt']})
        if error:
            return jsonify({"error": error}), 500

        asset_id = save_asset(generated_image)
        layer = {"id": len(layers)+1, "asset_id": asset_id, "position": {"x":0, "y":0}, "scale":1.0}
        layers.append(layer)
        save_layers()

        return jsonify({
            "image": base64.b64encode(generated_image).decode('utf-8'),
            "layer_id": layer["id"],
            "asset_id": asset_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/apply-mockup-advanced', methods=['POST'])
def apply_mockup_advanced():
    try:
        if 'base' not in request.files or 'design' not in request.files:
            return jsonify({"error": "Both base and design images are required."}), 400

        base_file = request.files['base']
        design_file = request.files['design']

        if base_file.filename == '' or design_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Load PIL images
        base_img = Image.open(base_file.stream).convert("RGB")
        design_img = Image.open(design_file.stream).convert("RGBA").resize(base_img.size)

        # Convert to NumPy arrays (this is required before any math operations)
        base_np = np.array(base_img)
        design_np = np.array(design_img)

        if model is None:
            load_models()
            
        # Compute the depth map from the base image
        depth_map = estimate_depth(base_img)
        # Normalize and blur the depth map for smoother results
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

        # Compute displacement fields
        map_x, map_y = compute_displacement_field(depth_map, 15.0)
        
        # Warp the design image using the displacement fields
        warped_design = warp_design_image(design_np, map_x, map_y)

        # Apply surface texture effects
        composite = apply_surface_texture(warped_design, base_np, depth_map)

        # Convert back to uint8 for saving
        composite = composite.astype(np.uint8)
        
        # Return the image
        _, buffer = cv2.imencode('.png', composite)
        return send_file(BytesIO(buffer.tobytes()), mimetype='image/png')
    
    except Exception as e:
        app.logger.error(f"Mockup error: {str(e)}")
        return jsonify({"error": f"Error processing mockup: {str(e)}"}), 500

@app.route('/layers', methods=['GET'])
def get_layers():
    return jsonify([{
        "id": l['id'],
        "asset_id": l.get('asset_id'),
        "position": l['position'],
        "scale": l['scale']
    } for l in layers])

@app.route('/layers/<int:layer_id>', methods=['PUT'])
def update_layer(layer_id):
    data = request.json
    for layer in layers:
        if layer['id'] == layer_id:
            layer.update({k: v for k, v in data.items() if k in ['position', 'scale']})
            save_layers()
            return jsonify(layer)
    return jsonify({"error": "Layer not found"}), 404

@app.route('/assets', methods=['GET'])
def get_assets():
    return jsonify([{"id": i, "preview": f"/asset/{i}"} for i in range(len(assets))])

@app.route('/asset/<int:asset_id>', methods=['GET'])
def get_asset(asset_id):
    if asset_id < len(assets):
        return send_file(BytesIO(assets[asset_id]), mimetype='image/png')
    else:
        return jsonify({"error": "Asset not found"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "layers": len(layers), "assets": len(assets)})

if __name__ == '__main__':
    ensure_data_dirs()
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)