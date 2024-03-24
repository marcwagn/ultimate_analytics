from tasks import flask_app, long_running_task
from celery.result import AsyncResult
from flask import request,jsonify 

@flask_app.post("/trigger_task")
def start_task() -> dict[str, object]:
    iterations = request.args.get('iterations')
    print(iterations)
    result = long_running_task.delay(int(iterations))
    return {"result_id": result.id}

@flask_app.get("/get_result")
def task_result() -> dict[str, object]:
    result_id = request.args.get('result_id')
    result = AsyncResult(result_id)
    if result.ready():
        # Task has completed
        if result.successful():
    
            return {
                "ready": result.ready(),
                "successful": result.successful(),
                "value": result.result,
            }
        else:
        # Task completed with an error
            return jsonify({'status': 'ERROR', 'error_message': str(result.result)})
    else:
        # Task is still pending
        return jsonify({'status': 'Running'})

if __name__ == "__main__":
    flask_app.run(debug=True)