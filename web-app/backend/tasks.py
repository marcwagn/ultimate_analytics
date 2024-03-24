from config import create_app
from celery import shared_task 
from time import sleep

flask_app = create_app()
celery_app = flask_app.extensions["celery"]

@shared_task(ignore_result=False)
def long_running_task(iterations) -> int:
    result = 0
    for i in range(iterations):
        result += i
        sleep(2) 
    return result