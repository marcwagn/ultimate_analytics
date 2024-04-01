from typing import Any, Callable
from celery import shared_task
from celery import Task
from celery.utils.log import get_task_logger
from .yolo_helper import make_callback_adapter_with_counter, convert_tracking_results_to_pandas
from .keypoints import KeypointsExtractor
from .homography import calculate_homography_matrix, convert_h
from .precalculated import get_precalculated_predictions_if_present
import ultralytics
import cv2
import os
import pandas as pd
import numpy as np
import torch

logger = get_task_logger(__name__)

@shared_task(bind=True, ignore_result=False)
def video_analysis(self: Task, video_path: str) -> object:
    logger.info(f"Start analysis for video: {video_path}")
    self.update_state(state="PROGRESS", meta={"status": 0})

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Found {total_frames} frames in video: {video_path}")

    def update_progressbar(frame):
        logger.info(f"YOLO object tracking for {video_path}: frame {frame}")
        self.update_state(state="PROGRESS", meta={"status": frame / total_frames })

    model_dir = os.getenv("MODEL_DATA_DIR", "data/model")
    model_path = os.path.join(model_dir, "best.pt")

    tracking_results = get_precalculated_predictions_if_present(video_path) \
        or _track(model_path=model_path, video_path=video_path, progressbar_callback=update_progressbar)

    # Keypoints and perspective removal
    logger.info(f"Running YOLO detection and removing perspective from video {video_path}")
    tracking_results_df = _translate_coordinates(tracking_results, total_frames=total_frames)

    # TODO - team detection
    tracking_results_df["team"] = 0

    logger.info(f"Preparing final results for video {video_path}")
    tracking_results_dict = _convert_to_final_results(tracking_results_df)

    logger.info(f"Finished analysis for for video {video_path}")
    return {"status": tracking_results_dict }

def _track(model_path: str, video_path: str, progressbar_callback: Callable) -> Any:
        """
        Perform object tracking on the video with YOLOv8.
        Args:
            progressbar_callback (Callable[int]): a callback accepting 1 argument (frame number)
        Return:
            YOLO tracking results
        """

        model = ultralytics.YOLO(model_path, verbose=True)
        if progressbar_callback is not None and isinstance(progressbar_callback, Callable):
            yolo_progress_reporting_event = "on_predict_batch_start"
            progress_callback_wrapped = make_callback_adapter_with_counter(yolo_progress_reporting_event, 
                                                                        lambda _,counter: progressbar_callback(counter))
            model.add_callback(yolo_progress_reporting_event, progress_callback_wrapped)

        # NB - if torch package is installed in the CPU variant, the device will default to "cpu"
        device = 0 if torch.cuda.is_available() else "cpu" 
        tracking_results = model.track(source=video_path, agnostic_nms=True, show=False, device=device, stream=True)

        return tracking_results

def _translate_coordinates(tracking_results: list[ultralytics.engine.results.Results], total_frames: int) -> pd.DataFrame:
    tracking_results_df = convert_tracking_results_to_pandas(tracking_results)

    keypoints_extractor = KeypointsExtractor(tracking_results_df, conf_threshold=0.5, max_lookback=60)
    # Calculate homography matrices
    H_all = []
    for frame in range(0, total_frames):
        kps = keypoints_extractor.get_4_best_keypoint_pairs(frame)
        if kps is None:
            raise RuntimeError(f"Error: couldnt detect enough keypoints for frame {frame}")
        h = calculate_homography_matrix(kps)
        H_all.append(h)
        
    def remove_perspective(row):
         h = H_all[row["frame"]]
         if h is None:
              row["x"] = np.nan
              row["y"] = np.nan
         else:
            translated_coords = convert_h(h, row["x":"y"].to_numpy())
            row["x"] = translated_coords[0]
            row["y"] = translated_coords[1]
         return row

    tracking_results_df = tracking_results_df.apply(remove_perspective, axis=1)
    return tracking_results_df

def _convert_to_final_results(tracking_results_df: pd.DataFrame) -> dict:
    converted_results = tracking_results_df[tracking_results_df["cls"].isin([0, 29, 30])]
    converted_results = converted_results[["cls", "x", "y", "team", "id", "frame"]]
    # NB - video frames start typically from 1
    converted_results["frame"] = converted_results["frame"] + 1
    converted_results = converted_results.dropna()

    final_dict = {}
    for frame in converted_results["frame"].unique(): 
        results_for_frame = converted_results[converted_results["frame"]==frame].drop(columns=["frame"])
        # NB - We convert frame number to string so that it can be later JSONified correctly as a dictionary key
        final_dict[str(frame)] = results_for_frame.to_dict(orient="records")
         
    return final_dict

