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
from packaging import version
import skimage
import json
import torch  # For depth estimation with MiDaS
import torchvision.transforms.functional as TF  # To convert PIL image to tensor

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'

# API endpoints for external services
BG_REMOVAL_API = "https://hf.space/embed/not-lain/background-removal/api/predict/"
TEXT_TO_IMAGE_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

if not os.getenv('HUGGINGFACE_API_TOKEN'):
    print("WARNING: HUGGINGFACE_API_TOKEN not set in environment variables")

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN', '')}"
}

# File-based storage for persistence
LAYERS_FILE = 'data/layers.json'
ASSETS_FOLDER = 'data/assets'
layers = []
assets = []

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
    for asset_file in asset_files:
        with open(os.path.join(ASSETS_FOLDER, asset_file), 'rb') as f:
            assets.append(f.read())

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
        local_headers = HEADERS.copy()
        if "json" in kwargs:
            local_headers["Content-Type"] = "application/json"
        response = requests.post(api_url, headers=local_headers, **kwargs, timeout=30)
        if response.status_code == 503:
            return None, "Model is loading, please try again in a moment"
        elif response.status_code != 200:
            return None, f"API Error: {response.status_code} - {response.text}"
        return response.content, None
    except requests.exceptions.Timeout:
        return None, "Request timed out. The server might be under heavy load."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Please check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# -------------------- Depth Estimation & Warping Functions --------------------

def load_midas_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transforms

midas_model, midas_transforms = load_midas_model()

def estimate_depth(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    input_batch = midas_transforms(pil_img)
    if isinstance(input_batch, Image.Image):
        input_batch = TF.to_tensor(np.array(input_batch))
    elif not torch.is_tensor(input_batch):
        input_batch = TF.to_tensor(input_batch)
    input_batch = input_batch.unsqueeze(0)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=pil_img.size[::-1],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

def compute_displacement_field(depth_map, displacement_strength=10.0):
    norm_depth = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    grad_x = cv2.Sobel(norm_depth, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(norm_depth, cv2.CV_32F, 0, 1, ksize=5)
    h, w = norm_depth.shape
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32) + grad_x * displacement_strength
    map_y = map_y.astype(np.float32) + grad_y * displacement_strength
    return map_x, map_y

def warp_design_image(design_img, map_x, map_y):
    warped_design = cv2.remap(design_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return warped_design

# -------------------- API Endpoints --------------------

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

        try:
            Image.open(BytesIO(image_bytes))
        except Exception:
            return jsonify({"error": "Invalid image format"}), 400

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"data": [image_b64]}
        processed_image, error = query_huggingface_api(BG_REMOVAL_API, json=payload)
        if error:
            return jsonify({"error": error}), 500

        asset_id = save_asset(processed_image)
        layer_id = len(layers) + 1
        layer = {
            "id": layer_id,
            "asset_id": asset_id,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        }
        layers.append(layer)
        save_layers()

        return jsonify({
            "image": base64.b64encode(processed_image).decode("utf-8"),
            "layer_id": layer_id,
            "asset_id": asset_id
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        payload = {"inputs": data['prompt']}
        generated_image, error = query_huggingface_api(TEXT_TO_IMAGE_API, json=payload)
        if error:
            return jsonify({"error": error}), 500
        
        img = Image.open(BytesIO(generated_image))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        processed_image = img_byte_arr.getvalue()

        asset_id = save_asset(processed_image)
        layer_id = len(layers) + 1
        layer = {
            "id": layer_id,
            "asset_id": asset_id,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        }
        layers.append(layer)
        save_layers()

        return jsonify({
            "image": base64.b64encode(processed_image).decode('utf-8'),
            "layer_id": layer_id,
            "asset_id": asset_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/apply-mockup-advanced', methods=['POST'])
def apply_mockup_advanced():
    """
    Apply a design to a mockup using advanced image processing.
    If perspective points are provided, use homography; otherwise,
    use depth estimation and displacement warping.
    """
    try:
        print("Starting mockup processing...")
        if 'base' not in request.files or 'design' not in request.files:
            return jsonify({"error": "Both base and design images are required."}), 400

        base_file = request.files['base']
        design_file = request.files['design']
        base_bytes = base_file.read()
        design_bytes = design_file.read()

        base_img = cv2.imdecode(np.frombuffer(base_bytes, np.uint8), cv2.IMREAD_COLOR)
        if base_img is None:
            print("Error: Could not decode base image.")
            return jsonify({"error": "Could not decode base image."}), 400

        design_img = cv2.imdecode(np.frombuffer(design_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if design_img is None:
            print("Error: Could not decode design image.")
            return jsonify({"error": "Could not decode design image."}), 400
        
        if len(design_img.shape) == 3 and design_img.shape[2] == 4:
            design_img = cv2.cvtColor(design_img, cv2.COLOR_BGRA2BGR)

        print("Base image shape:", base_img.shape)
        print("Design image shape:", design_img.shape)

        design_img = cv2.resize(design_img, (base_img.shape[1], base_img.shape[0]))
        print("Resized design image shape:", design_img.shape)

        perspective_points = request.form.get('perspective_points')
        if perspective_points:
            try:
                pts_dst = np.array(json.loads(perspective_points), dtype=np.float32)
                if pts_dst.shape != (4, 2):
                    raise ValueError("Perspective points must be a 4x2 array")
            except (json.JSONDecodeError, ValueError) as e:
                return jsonify({"error": f"Invalid perspective points: {str(e)}"}), 400
        else:
            pts_dst = None

        if pts_dst is not None:
            print("Using homography-based warping...")
            h_design, w_design = design_img.shape[:2]
            pts_src = np.array([[0, 0], [w_design, 0], [w_design, h_design], [0, h_design]], dtype=np.float32)
            M, status = cv2.findHomography(pts_src, pts_dst)
            warped_design = cv2.warpPerspective(design_img, M, (base_img.shape[1], base_img.shape[0]))
        else:
            print("Using depth estimation and displacement warping...")
            try:
                pil_base = Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
                depth_map = estimate_depth(pil_base)
                print("Depth map shape:", depth_map.shape)
                map_x, map_y = compute_displacement_field(depth_map)
                warped_design = warp_design_image(design_img, map_x, map_y)
            except Exception as depth_err:
                print("Depth warping failed:", depth_err)
                print("Falling back to simple composite without warping.")
                warped_design = design_img

        try:
            warped_design_float = warped_design.astype(np.float32)
            base_img_float = base_img.astype(np.float32)
            if version.parse(skimage.__version__) >= version.parse("0.19"):
                matched_design = match_histograms(warped_design_float, base_img_float, channel_axis=-1)
            else:
                matched_design = match_histograms(warped_design_float, base_img_float, multichannel=True)
            matched_design = np.clip(matched_design, 0, 255).astype(np.uint8)
            print("Histogram matching succeeded.")
        except Exception as hist_err:
            print("Histogram matching failed:", hist_err)
            matched_design = warped_design

        alpha = 0.7
        try:
            composite = cv2.addWeighted(matched_design, alpha, base_img, 1 - alpha, 0)
        except Exception as add_err:
            print("Error during compositing:", add_err)
            # Fall back to overlaying design on base
            composite = base_img
        print("Composite image created.")

        retval, buffer = cv2.imencode('.png', composite)
        composite_bytes = buffer.tobytes()
        
        asset_id = save_asset(composite_bytes)
        layer_id = len(layers) + 1
        layer = {
            "id": layer_id,
            "asset_id": asset_id,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        }
        layers.append(layer)
        save_layers()
        
        print("Mockup processing complete. Returning composite image.")
        return send_file(BytesIO(composite_bytes), mimetype='image/png', download_name='advanced_mockup.png')
    
    except Exception as e:
        print("Error in apply-mockup-advanced:", str(e))
        return jsonify({"error": f"Error processing mockup: {str(e)}"}), 500

@app.route('/layers', methods=['GET'])
def get_layers():
    return jsonify([{
        "id": layer['id'],
        "asset_id": layer.get('asset_id'),
        "position": layer['position'],
        "scale": layer['scale']
    } for layer in layers])

@app.route('/layers/<int:layer_id>', methods=['PUT'])
def update_layer(layer_id):
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    for i, layer in enumerate(layers):
        if layer['id'] == layer_id:
            if 'position' in data:
                layers[i]['position'] = data['position']
            if 'scale' in data:
                layers[i]['scale'] = data['scale']
            save_layers()
            return jsonify({"success": True, "layer": layers[i]})
    
    return jsonify({"error": "Layer not found"}), 404

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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "layers": len(layers), "assets": len(assets)})

if __name__ == '__main__':
    ensure_data_dirs()
    app.run(host='0.0.0.0', port=5000, debug=True)
