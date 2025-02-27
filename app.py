import os
import requests
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_file, render_template
from dotenv import load_dotenv
from PIL import Image
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'

BG_REMOVAL_API = "https://api-inference.huggingface.co/models/not-lain/background-removal"
TEXT_TO_IMAGE_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"
}

# In-memory storage for layers and assets (replace with database in production)
layers = []
assets = []

def query_huggingface_api(api_url, **kwargs):
    """
    Send a POST request to the given Hugging Face API endpoint.
    Accepts keyword arguments so you can pass either raw data or json.
    """
    response = requests.post(api_url, headers=HEADERS, **kwargs)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    return response.content

# Root route renders the index.html
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    """Handle background removal using Hugging Face model"""
    try:
        if 'file' in request.files:
            # Handling file upload from form
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            image_bytes = file.read()
        elif request.is_json and 'image' in request.json:
            # Handling base64 encoded image
            image_bytes = base64.b64decode(request.json['image'])
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Process image through Hugging Face API
        processed_image = query_huggingface_api(BG_REMOVAL_API, data=image_bytes)
        
        # Create new layer
        layer_id = len(layers) + 1
        layers.append({
            "id": layer_id,
            "image": processed_image,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        })

        # Add to assets
        assets.append(processed_image)

        # Return either raw image data or base64 depending on request format
        if request.is_json:
            return jsonify({"image": base64.b64encode(processed_image).decode('utf-8')})
        else:
            return send_file(
                BytesIO(processed_image),
                mimetype='image/png',
                download_name=f'layer_{layer_id}.png'
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Generate image from text prompt using Hugging Face model"""
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Generate image through Hugging Face API using JSON payload
        payload = {"inputs": data['prompt']}
        generated_image = query_huggingface_api(TEXT_TO_IMAGE_API, json=payload)
        
        # Convert to PNG
        img = Image.open(BytesIO(generated_image))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        processed_image = img_byte_arr.getvalue()

        # Create new layer
        layer_id = len(layers) + 1
        layers.append({
            "id": layer_id,
            "image": processed_image,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        })

        # Add to assets
        assets.append(processed_image)

        return jsonify({"image": base64.b64encode(processed_image).decode('utf-8')})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/layers', methods=['GET'])
def get_layers():
    """Get all layers"""
    return jsonify([{
        "id": layer['id'],
        "position": layer['position'],
        "scale": layer['scale']
    } for layer in layers])

@app.route('/assets', methods=['GET'])
def get_assets():
    """Get all assets"""
    return jsonify([{
        "id": idx,
        "preview": f"/asset/{idx}"
    } for idx in range(len(assets))])

@app.route('/asset/<int:asset_id>', methods=['GET'])
def get_asset(asset_id):
    """Get specific asset"""
    if asset_id >= len(assets):
        return jsonify({"error": "Asset not found"}), 404
    return send_file(
        BytesIO(assets[asset_id]),
        mimetype='image/png'
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)