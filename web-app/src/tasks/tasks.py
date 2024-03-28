from celery import shared_task
from celery import Task
from celery.utils.log import get_task_logger
from yolo_helper import make_callback_adapter_with_counter, convert_tracking_results_to_pandas
from src.keypoints import KeypointsExtractor, KeypointQuad
from src.homography import calculate_homography_matrix, convert_h
from src.video_object import VideoObject
from src.team_detector import TeamDetector

logger = get_task_logger(__name__)


@shared_task(bind=True, ignore_result=False)
def video_analysis(self: Task, video_path: str) -> object:
    logger.info(f"Start analysis for video: {video_path}")
    self.update_state(state="PROGRESS", meta={"status": 0})

    video_object = VideoObject(video_path=video_path)

    #video = cv2.VideoCapture(video_path)
    #total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = video_object.get_num_frames()

    def update_progressbar(frame):
        if frame % 100:
            logger.info(f"YOLO object tracking for {video_path}: frame {frame}")
        self.update_state(state="PROGRESS", meta={"status": frame / total_frames })

    model_path = "TODO"
    tracking_results = track(self, model_path=model_path, video_path=video_path, progressbar_callback=update_progressbar)

    # Team detection
    for frame in range(0, total_frames):
        img = video_object.get_frame(frame)
        team_detector = TeamDetector(img=img, detector_results=tracking_results)
        team_prediction = team_detector.predict_player_clusters(player_imgs=TODO)

    # Keypoints and perspective removal
    tracking_results_df = convert_tracking_results_to_pandas(tracking_results)

    keypoints_extractor = KeypointsExtractor(tracking_results_df, conf_threshold=0.5, max_lookback=30)
    H_all = [calculate_homography_matrix(keypoints_extractor.get_4_best_keypoint_pairs(frame)) for frame in range(0, total_frames)]
    # TODO - make it in 1 pass
    tracking_results_df["x_t"] = tracking_results_df.apply(lambda row: convert_h(H_all[row["frame"]], row[1:3].to_numpy())[0], axis=1)
    tracking_results_df["y_t"] = tracking_results_df.apply(lambda row: convert_h(H_all[row["frame"]], row[1:3].to_numpy())[1], axis=1)


    # for frame in range(0,total_frames,100):
    #     #logger.info(f"Detected frames: {frames}")
    #     time.sleep(1)

     # TODO
    tracking_results_dict = tracking_results_df
    return {"status": tracking_results_dict }

def track(self, model_path: str, video_path: str, progressbar_callback: Callable) -> Any
        """
        Perform object tracking on the video with YOLOv8.
        Args:
            progressbar_callback (Callable[int]): a callback accepting 1 argument (frame number)
        Return:
            YOLO tracking results
        """
        model = ultralytics.YOLO(model_path, verbose=True)
        yolo_progress_reporting_event = "on_predict_batch_start"
        progress_callback_wrapped = make_callback_adapter_with_counter(yolo_progress_reporting_event, 
                                                                       lambda _,counter: progressbar_callback(counter))
        model.add_callback(yolo_progress_reporting_event, progress_callback_wrapped)

        device = 0 if torch.cuda.is_available() else 'cpu' 
        tracking_results = model.track(source=video_path, agnostic_nms=True, show=False, device=device, stream=True)

        return tracking_results
