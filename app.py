from flask import Flask, request, send_file, jsonify, render_template
import os
import requests
from zipfile import ZipFile
from io import BytesIO
import io
import tempfile
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    data_json = request.json

    input_text = data_json.get('input')
    negative_prompt = data_json.get('negative_prompt')
    width = int(data_json.get('width', 1216))
    height = int(data_json.get('height', 832))
    steps = int(data_json.get('steps', 30))
    guidance_scale = float(data_json.get('guidance_scale', 5))

    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {data_json.get("key")}',
        'Content-Type': 'application/json',
    }

    payload_data = {
        "input": input_text,
        "model": "nai-diffusion-3",
        "action": "generate",
        "parameters": {
            "width": width,
            "height": height,
            "steps": steps,
            "sampler": "k_euler",
            "guidance": guidance_scale,
        },
    }

    if negative_prompt:
        payload_data['parameters']['negative_prompt'] = negative_prompt

    response_api = requests.post(
            url='https://api.novelai.net/ai/generate-image',
            headers=headers,
            json=payload_data)

    if response_api.status_code == 200:
        zip_file_path = os.path.join(tempfile.mkdtemp(), 'image.zip')
        with open(zip_file_path, 'wb') as f_zip:
            f_zip.write(response_api.content)

        return handle_and_send_image(zip_file_path)
    else:
        return jsonify({"error": response_api.text}), response_api.status_code

def handle_and_send_image(zip_file_path):
    output_directory = os.path.join(os.getcwd(), 'output')

    with ZipFile(zip_file_path) as zip_ref:
        zip_ref.extractall(output_directory)

    current_files = [f for f in os.listdir(output_directory) if f.startswith("image_") and f.endswith(".png")]

    highest_num = max([int(f[6:-4]) for f in current_files], default=-1) + 1

    new_filename = f'image_{highest_num}.png'

    extracted_filepath = os.path.join(output_directory, zip_ref.namelist()[0])

    final_filepath = os.path.join(output_directory, new_filename)

    os.rename(extracted_filepath, final_filepath)

    os.remove(zip_file_path)

    with open(final_filepath, 'rb') as img:
        return send_file(
            io.BytesIO(img.read()),
            mimetype='image/png',
            as_attachment=True,
        )

@app.route('/latest-image')
def latest_image():
    output_directory = os.path.join(os.getcwd(), 'output')
    current_files=[f for f in os.listdir(output_directory) if f.startswith("image_") and f.endswith(".png")]
    highest_num=max([int(f[6:-4]) for f in current_files], default=-1)
    filename=f'image_{highest_num}.png'
    filepath=os.path.join(output_directory,filename)
    return send_file(filepath,mimetype='image/png',as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=True)