import time

from celery import shared_task
from celery import Task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@shared_task(bind=True, ignore_result=False)
def upload(self: Task, video_path: str) -> object:
    logger.info(video_path)

    #run some task
    total = 100
    for i in range(0,total,10):
        self.update_state(state="PROGRESS", meta={"current": i + 1, "total": total})
        time.sleep(1)

    return {"current": total, "total": total}