from flask import Flask, request, jsonify
from flask import render_template
from celery.result import AsyncResult
from flask_cors import CORS

from .import tasks

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

    result = tasks.video_analysis.delay(video_path=video_path)

    return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/result/<id>', methods=['GET'])  
def result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    ready = result.ready()
    return {
        "ready": ready,
        "successful": result.successful() if ready else None,
        "value": result.get() if ready else result.result,
    }

if __name__ == '__main__':
    app.run(debug=True)
    #, ssl_context='adhoc'