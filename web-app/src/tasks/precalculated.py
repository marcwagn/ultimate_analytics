
import os
import pickle
from typing import Union
import ultralytics
import hashlib
from pathlib import Path
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

def get_precalculated_predictions_if_present(video_path: str) -> Union[list[ultralytics.engine.results.Results], None]:
    precalculated_dir = os.getenv("PRECALCULATED_DATA_DIR")
    if precalculated_dir is None or precalculated_dir == "":
         logger.info("PRECALCULATED_DATA_DIR not set, skipping")
         return None
    
    video_sha256sum = _calculate_sha256sum(video_path)
    maybe_pickled_results_path = Path(precalculated_dir)/(video_sha256sum + ".pickle")
    if os.path.exists(maybe_pickled_results_path) and os.path.isfile(maybe_pickled_results_path):
        with open(maybe_pickled_results_path, "rb") as f:
            tracking_results = pickle.load(f)
            return tracking_results
    logger.info(f"Did not find {maybe_pickled_results_path.name} in {precalculated_dir}")
    return None
    
    
def _calculate_sha256sum(filename):
    with open(filename, 'rb', buffering=0) as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()