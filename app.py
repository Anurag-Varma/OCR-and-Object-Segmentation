# Import necessary modules and functions from Flask, Werkzeug, and other libraries
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import os
# Import modules related to segmentation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.transforms import InterpolationMode
import base64
import requests
import shutil

# Initialize Flask application
app = Flask(__name__)

# Constants and configurations
BICUBIC = InterpolationMode.BICUBIC
device = "cpu"
image_path = "cheezit_1.png"  # Path to the default image
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Checkpoint for the segmentation model
model_type = "vit_h"  # Type of the segmentation model
api_key = 'OpenAI_API_key'  # OpenAI API key

# Initialize segmentation-related objects
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=9,
    pred_iou_thresh=0.05,
    stability_score_thresh=0.95,
    min_mask_region_area=1000,  
)

# Function to clear a directory
def clear_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

# Function to encode an image to base64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Configuration for file upload
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure that the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # Clear the directory where extracted objects are stored
    clear_directory('static/extracted_objects/')
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # Check if the file is empty
    if file.filename == '':
        return redirect(request.url)
    # If the file is valid, save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Open the uploaded image and convert it to numpy array
        pil_img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_np = np.array(pil_img)

        # Generate masks for segmentation
        masks = mask_generator.generate(image_np)

        # Filter out small masks
        masks = list(filter(lambda m: m["area"] > 1000, masks))

        # Create a directory for storing extracted objects
        output_dir = "static/extracted_objects/"
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through masks, extract objects, and save them as separate images
        for i, mask in enumerate(masks):
            segmentation = mask['segmentation']
            object_image = Image.new("RGBA", pil_img.size)
            object_data = np.array(object_image)

            original_data = np.dstack([image_np, np.ones(image_np.shape[:2], dtype=np.uint8) * 255])

            object_data[segmentation] = original_data[segmentation]
            
            object_image = Image.fromarray(object_data)

            output_path = os.path.join(output_dir, f"object_{i+1}.png")
            object_image.save(output_path)

        print("All objects have been extracted and saved.")

        # Encode the uploaded image to base64 format for OpenAI API
        base64_image = encode_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Prepare headers and payload for OpenAI API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Identify the exact text present in the given image and return it in meaningful texts in new line separated by \n Don't give any other extra text or new words. If anything is repeated, then give it only once."
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
        }

        # Send request to OpenAI API
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Extract content from API response
        content = response.json()['choices'][0]['message']['content']

        # Split content into lines
        text_list = []

        for line in content.split('\n'):
            text_list.append(line)

        # Get paths of extracted object images
        EXTRACTED_FOLDER = 'static/extracted_objects/'
        image_paths = [os.path.join(EXTRACTED_FOLDER, f) for f in os.listdir(EXTRACTED_FOLDER) if allowed_file(f)]
        # Generate URLs for displaying images on the webpage
        images = [url_for('static', filename='extracted_objects/' + os.path.basename(path)) for path in image_paths]
        
        # Render the index.html template with the extracted images and text
        return render_template('index.html', images=image_paths, text_list=text_list)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
