import time
import cv2

from celery import shared_task
from celery import Task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@shared_task(bind=True, ignore_result=False)
def video_analysis(self: Task, video_path: str) -> object:
    logger.info(f"Start analyis for video: {video_path}")
    self.update_state(state="PROGRESS", meta={"status": 0})

    video = cv2.VideoCapture(video_path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in range(0,frames,100):
        self.update_state(state="PROGRESS", meta={"status": frame / frames })
        
        
        #logger.info(f"Detected frames: {frames}")
        time.sleep(1)

    return {"status": 100}