from celery.result import AsyncResult
from flask import Blueprint
from flask import request, jsonify

from tasks import tasks

import os
import tempfile

bp = Blueprint("tasks", __name__, url_prefix="/tasks")

@bp.get("/result/<id>")
def result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    ready = result.ready()
    return {
        "ready": ready,
        "successful": result.successful() if ready else None,
        "value": result.get() if ready else result.result,
    }

@bp.post('/upload')
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, file.filename)

    file.save(video_path)
    print(f"File samed to {video_path}")

    result = tasks.video_analysis.delay(video_path=video_path)

    return {"result_id": result.id}