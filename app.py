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

@app.route('/combine-images', methods=['POST'])
def combine_images():
    """Combine multiple images with their transformations applied"""
    try:
        data = request.json
        if not data or 'layers' not in data:
            return jsonify({"error": "No layers provided"}), 400
        
        # Create a blank canvas with the specified dimensions
        width = data.get('width', 1200)
        height = data.get('height', 800)
        background_color = data.get('backgroundColor', (0, 0, 0, 0))  # Transparent by default
        
        # Create a new image with transparency
        combined_image = Image.new('RGBA', (width, height), background_color)
        
        # Process each layer
        for layer_data in data['layers']:
            # Skip if layer is not visible
            if not layer_data.get('visible', True):
                continue
                
            # Get the layer image data
            if 'imageData' in layer_data:
                # Image data provided as base64
                layer_image_bytes = base64.b64decode(layer_data['imageData'])
                layer_image = Image.open(BytesIO(layer_image_bytes))
            elif 'assetId' in layer_data and layer_data['assetId'] < len(assets):
                # Use an asset from our stored assets
                layer_image = Image.open(BytesIO(assets[layer_data['assetId']]))
            else:
                continue
                
            # Apply transformations
            # 1. Resize
            original_width, original_height = layer_image.size
            new_width = int(original_width * layer_data.get('scale', 1.0))
            new_height = int(original_height * layer_data.get('scale', 1.0))
            if new_width != original_width or new_height != original_height:
                layer_image = layer_image.resize((new_width, new_height), Image.LANCZOS)
                
            # 2. Rotate (if rotation is provided)
            if 'rotation' in layer_data and layer_data['rotation'] != 0:
                layer_image = layer_image.rotate(
                    -float(layer_data['rotation']) * (180/3.14159),  # Convert radians to degrees
                    expand=True,
                    resample=Image.BICUBIC
                )
                
            # 3. Position (centered at the specified coordinates)
            position_x = layer_data.get('x', width/2)
            position_y = layer_data.get('y', height/2)
            paste_x = int(position_x - layer_image.width/2)
            paste_y = int(position_y - layer_image.height/2)
            
            # If the layer has transparency, we need to use alpha compositing
            if layer_image.mode == 'RGBA':
                # Create a temporary transparent image for this layer
                temp = Image.new('RGBA', combined_image.size, (0, 0, 0, 0))
                temp.paste(layer_image, (paste_x, paste_y), layer_image)
                combined_image = Image.alpha_composite(combined_image, temp)
            else:
                # Convert to RGBA to ensure compatibility
                layer_image = layer_image.convert('RGBA')
                temp = Image.new('RGBA', combined_image.size, (0, 0, 0, 0))
                temp.paste(layer_image, (paste_x, paste_y), layer_image)
                combined_image = Image.alpha_composite(combined_image, temp)
        
        # Convert the final image to PNG
        output = BytesIO()
        combined_image.save(output, format='PNG')
        output.seek(0)
        
        # Return the combined image
        return send_file(
            output,
            mimetype='image/png',
            download_name='combined_image.png'
        )
        
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