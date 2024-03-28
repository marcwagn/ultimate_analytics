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

    coordinates = {}

    for frame in range(frames):
        if frame % 100 == 0:
            logger.info(f"Processing frame {frame}")

        coordinates[frame] = [  { "cls": 0, "x": 5, "y": 15, "team": 0, "id": 0 },
                                { "cls": 0, "x": 25, "y": 15, "team": 0, "id": 8 },
                                { "cls": 0, "x": 5, "y": 85, "team": 1, "id": 0 },
                                { "cls": 0, "x": 25, "y": 85, "team": 1, "id": 8 },
                                { "cls": 30, "x": 10, "y": 20, "id": 5 }]
        
        self.update_state(state="PROGRESS", meta={"status": frame / frames })

    return {"status": 100, "coordinates": coordinates}