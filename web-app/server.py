from flask import Flask, request, jsonify
from flask import render_template

from flask_cors import CORS
#from .config import create_app

import os
import tempfile

app = Flask(__name__)
CORS(app)

@app.route("/")
def index() -> str:
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, file.filename)

    file.save(video_path)

    #web worker [async]
    # scedule a task to process the video
    # 1. pulling approach [client ask server for the status of the video processing] 
    # 2. demo for websocket [duplex communication]

    print(f'File saved at {video_path}')

    return jsonify({'message': 'File uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True, ssl_context='adhoc')