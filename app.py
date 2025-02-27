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
app.config['MOCKUPS_FOLDER'] = 'mockups'

BG_REMOVAL_API = "https://api-inference.huggingface.co/models/not-lain/background-removal"
TEXT_TO_IMAGE_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"
}

# In-memory storage for layers and assets
layers = []
assets = []
mockups = []  # Storage for mockup templates

def query_huggingface_api(api_url, **kwargs):
    """
    Send a POST request to the given Hugging Face API endpoint.
    Accepts keyword arguments so you can pass either raw data or json.
    """
    response = requests.post(api_url, headers=HEADERS, **kwargs)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    return response.content

def init_mockups():
    """Initialize built-in mockups"""
    if not os.path.exists(app.config['MOCKUPS_FOLDER']):
        os.makedirs(app.config['MOCKUPS_FOLDER'], exist_ok=True)
        # You would add default mockups here in production
    
    # Load mockups from folder
    for filename in os.listdir(app.config['MOCKUPS_FOLDER']):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(app.config['MOCKUPS_FOLDER'], filename)
            with open(path, 'rb') as f:
                mockup_data = f.read()
                name = os.path.splitext(filename)[0]
                mockups.append({
                    "id": len(mockups),
                    "name": name,
                    "image": mockup_data,
                    "smart_object_area": {
                        # Default values - in production you would 
                        # have these saved in a JSON file or database
                        "x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5,
                        "perspective": "flat"  # flat, perspective, curved
                    }
                })

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
        elif request.is_json:
            # Handling base64 encoded image or selected layer
            if 'image' in request.json:
                image_bytes = base64.b64decode(request.json['image'])
            elif 'layerId' in request.json:
                # Get layer by ID (from in-memory storage)
                layer_id = request.json['layerId']
                layer = next((l for l in layers if l["id"] == layer_id), None)
                if not layer:
                    return jsonify({"error": f"Layer {layer_id} not found"}), 404
                image_bytes = layer["image"]
            else:
                return jsonify({"error": "No image or layer provided"}), 400
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
            return jsonify({
                "image": base64.b64encode(processed_image).decode('utf-8'),
                "layerId": layer_id
            })
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

        return jsonify({
            "image": base64.b64encode(processed_image).decode('utf-8'),
            "layerId": layer_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/apply-mockup', methods=['POST'])
def apply_mockup():
    """Apply a design to a mockup template"""
    data = request.json
    if not data or 'designLayerId' not in data or 'mockupId' not in data:
        return jsonify({"error": "Design layer ID and mockup ID required"}), 400
    
    try:
        # Get the design layer
        design_layer_id = data['designLayerId']
        design_layer = next((l for l in layers if l["id"] == design_layer_id), None)
        if not design_layer:
            return jsonify({"error": f"Design layer {design_layer_id} not found"}), 404
        
        # Get the mockup
        mockup_id = data['mockupId']
        if mockup_id >= len(mockups):
            return jsonify({"error": f"Mockup {mockup_id} not found"}), 404
        mockup = mockups[mockup_id]
        
        # Open images with PIL
        design_img = Image.open(BytesIO(design_layer["image"]))
        mockup_img = Image.open(BytesIO(mockup["image"]))
        
        # Apply design to mockup using smart object area
        smart_area = mockup["smart_object_area"]
        mockup_width, mockup_height = mockup_img.size
        
        # Calculate smart object area in pixels
        smart_x = int(smart_area["x"] * mockup_width)
        smart_y = int(smart_area["y"] * mockup_height)
        smart_width = int(smart_area["width"] * mockup_width)
        smart_height = int(smart_area["height"] * mockup_height)
        
        # Resize design to fit smart object area
        design_resized = design_img.resize((smart_width, smart_height))
        
        # If perspective or curved mockup, would apply transformation here
        # For this example, using simple paste
        mockup_img.paste(design_resized, (smart_x, smart_y), design_resized if design_resized.mode == 'RGBA' else None)
        
        # Convert to PNG
        result_bytes = BytesIO()
        mockup_img.save(result_bytes, format='PNG')
        result_image = result_bytes.getvalue()
        
        # Create new layer with the result
        layer_id = len(layers) + 1
        layers.append({
            "id": layer_id,
            "image": result_image,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        })
        
        # Add to assets
        assets.append(result_image)
        
        return jsonify({
            "image": base64.b64encode(result_image).decode('utf-8'),
            "layerId": layer_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mockups', methods=['GET'])
def get_mockups():
    """Get all available mockups"""
    return jsonify([{
        "id": mockup["id"],
        "name": mockup["name"],
        "preview": f"/mockup/{mockup['id']}/preview"
    } for mockup in mockups])

@app.route('/mockup/<int:mockup_id>/preview', methods=['GET'])
def get_mockup_preview(mockup_id):
    """Get mockup preview image"""
    if mockup_id >= len(mockups):
        return jsonify({"error": "Mockup not found"}), 404
    return send_file(
        BytesIO(mockups[mockup_id]["image"]),
        mimetype='image/png'
    )

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
    init_mockups()  # Initialize mockup templates
    app.run(host='0.0.0.0', port=5000, debug=True)