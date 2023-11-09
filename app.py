import arch
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
import cv2
import numpy as np
import glob
import os.path as osp
app = Flask(__name__)
CORS(app)

def predict(input_image_path, output_folder='results'):
    model_path = 'models/RRDB_ESRGAN_x4.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)

    print(f'Model path {model_path}. \nPredicting...')

    base = osp.splitext(osp.basename(input_image_path))[0]

    # read input image
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    output_path = osp.join(output_folder, f'{base}_rlt.png')
    cv2.imwrite(output_path, output)

    return output_path

@app.route('/uploadimage', methods=['GET', 'POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        # Save the uploaded file to the 'uploads' folder
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, secure_filename(file.filename))
        file.save(file_path)
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty file without a name
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Check if the file is allowed
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            # Save the uploaded file to a temporary folder
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, secure_filename(file.filename))
            file.save(file_path)

            # Make a prediction using the uploaded file
            result_path = predict(file_path)

            # Return the result path or other response as needed
            return jsonify({'result': result_path})

        else:
            return jsonify({'error': 'File extension not allowed'})

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)