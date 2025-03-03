import os
import requests
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from PIL import Image
from flask_cors import CORS
import json

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

LAYERS_FILE = 'data/layers.json'
ASSETS_FOLDER = 'data/assets'
layers = []
assets = []

# In-Context-LoRA API
LORA_API = "https://api-inference.huggingface.co/models/ali-vilab/In-Context-LoRA"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN', '')}",
    "Content-Type": "application/json"
}

def query_huggingface_api(prompt, logo_image=None, logo_desc=""):
    try:
        # Prepare payload for In-Context-LoRA
        payload = {
            "inputs": {
                "prompt": prompt,
                "logo_description": logo_desc,
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        }
        if logo_image:
            buffered = BytesIO()
            logo_image.save(buffered, format="PNG")
            payload["inputs"]["logo_image"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        response = requests.post(LORA_API, headers=HEADERS, json=payload, timeout=60)
        
        # If response is not OK, try to extract an error message
        if response.status_code != 200:
            try:
                error_msg = response.json().get("error", response.text)
            except Exception:
                error_msg = response.text
            return None, error_msg
        
        # Check if the response is JSON (e.g., containing a base64 image) or raw image bytes
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            result_json = response.json()
            if "generated_image" in result_json:
                image_data = base64.b64decode(result_json["generated_image"])
                return image_data, None
            else:
                return None, "Invalid response format: 'generated_image' key not found"
        else:
            # Assume binary image data is returned
            return response.content, None
    except Exception as e:
        return None, str(e)

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

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        logo_desc = request.form.get('logo_desc', '')
        logo_file = request.files.get('logo')
        
        logo_image = None
        if logo_file:
            logo_image = Image.open(logo_file.stream).convert("RGB")
        
        generated_image, error = query_huggingface_api(prompt, logo_image, logo_desc)
        if error:
            return jsonify({"error": error}), 500

        asset_id = save_asset(generated_image)
        layer = {"id": len(layers)+1, "asset_id": asset_id, "position": {"x": 0, "y": 0}, "scale": 1.0}
        layers.append(layer)
        save_layers()

        return jsonify({
            "image": base64.b64encode(generated_image).decode('utf-8'),
            "layer_id": layer["id"],
            "asset_id": asset_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    ensure_data_dirs()
    app.run(host='0.0.0.0', port=5000, debug=True)
