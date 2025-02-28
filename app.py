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

# Use the correct endpoint for the Space
BG_REMOVAL_API = "https://hf.space/embed/not-lain/background-removal/api/predict/"
TEXT_TO_IMAGE_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"
}

# In-memory storage for layers and assets (replace with database in production)
layers = []
assets = []

def query_huggingface_api(api_url, **kwargs):
    # If sending JSON, set the content type header
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
    """Handle background removal using the Hugging Face Space API"""
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

        # Encode the image in base64 and build the JSON payload
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"data": [image_b64]}
        
        # Call the API using a JSON payload
        processed_image = query_huggingface_api(BG_REMOVAL_API, json=payload)

        # Create new layer and asset (for in-memory storage)
        layer_id = len(layers) + 1
        layers.append({
            "id": layer_id,
            "image": processed_image,
            "position": {"x": 0, "y": 0},
            "scale": 1.0
        })
        assets.append(processed_image)

        # Return result as base64-encoded image
        return jsonify({"image": base64.b64encode(processed_image).decode("utf-8")})
    
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
    try:
        data = request.json
        if not data or 'layers' not in data:
            return jsonify({"error": "No layers provided"}), 400
        
        width = data.get('width', 1200)
        height = data.get('height', 800)
        background_color = data.get('backgroundColor', (0, 0, 0, 0))
        
        mode = data.get("mode", "combine")
        
        if mode == "mockup":
            # Expect exactly two layers: base and design
            if len(data['layers']) < 2:
                return jsonify({"error": "Two layers required for mockup"}), 400
            base_layer = data['layers'][0]
            design_layer = data['layers'][1]
            if 'imageData' in base_layer:
                base_image_bytes = base64.b64decode(base_layer['imageData'])
                base_img = Image.open(BytesIO(base_image_bytes)).convert("RGBA")
            else:
                return jsonify({"error": "Base layer image data missing"}), 400
            if 'imageData' in design_layer:
                design_image_bytes = base64.b64decode(design_layer['imageData'])
                design_img = Image.open(BytesIO(design_image_bytes)).convert("RGBA")
            else:
                return jsonify({"error": "Design layer image data missing"}), 400
            
            # Resize design image to match base image dimensions if they differ
            if base_img.size != design_img.size:
                design_img = design_img.resize(base_img.size, Image.LANCZOS)
            
            # Create a mask from the base image (use alpha channel if available, else grayscale)
            try:
                mask = base_img.split()[3]
            except Exception:
                mask = base_img.convert("L")
            
            # Composite the design image over the base using the mask
            combined_img = Image.composite(design_img, base_img, mask)
        
        else:
            # Regular combination for multiple layers
            combined_img = Image.new('RGBA', (width, height), background_color)
            for layer_data in data['layers']:
                if not layer_data.get('visible', True):
                    continue
                if 'imageData' in layer_data:
                    layer_image_bytes = base64.b64decode(layer_data['imageData'])
                    layer_image = Image.open(BytesIO(layer_image_bytes)).convert("RGBA")
                elif 'assetId' in layer_data:
                    # Assuming assets is defined in memory
                    layer_image = Image.open(BytesIO(assets[layer_data['assetId']])).convert("RGBA")
                else:
                    continue
                
                original_width, original_height = layer_image.size
                scale = layer_data.get('scale', 1.0)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                if new_width != original_width or new_height != original_height:
                    layer_image = layer_image.resize((new_width, new_height), Image.LANCZOS)
                
                if 'rotation' in layer_data and layer_data['rotation'] != 0:
                    layer_image = layer_image.rotate(-float(layer_data['rotation']) * (180/3.14159),
                                                     expand=True, resample=Image.BICUBIC)
                
                pos_x = layer_data.get('x', width/2)
                pos_y = layer_data.get('y', height/2)
                paste_x = int(pos_x - layer_image.width/2)
                paste_y = int(pos_y - layer_image.height/2)
                temp = Image.new('RGBA', combined_img.size, (0, 0, 0, 0))
                temp.paste(layer_image, (paste_x, paste_y), layer_image)
                combined_img = Image.alpha_composite(combined_img, temp)
        
        output = BytesIO()
        combined_img.save(output, format='PNG')
        output.seek(0)
        return send_file(output, mimetype='image/png', download_name='combined_image.png')
    
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