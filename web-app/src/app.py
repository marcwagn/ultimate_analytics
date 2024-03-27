from celery import Celery, Task
from flask import Flask
from flask import render_template

from flask_cors import CORS
from dotenv import load_dotenv
import logging
import os
from google.cloud import logging as gcp_logging

def create_app() -> Flask:
    load_dotenv()
    if os.getenv("GCP_LOGGING") == "True":
        gcp_logging_client = gcp_logging.Client()
        gcp_logging_client.setup_logging()

    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    CORS(app)
    app.config.from_mapping(
        CELERY=dict(
            broker_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            result_backend=os.getenv("REDIS_URL", "redis://localhost:6379"),
            task_ignore_result=True,
        ),
    )
    app.config.from_prefixed_env()
    celery_init_app(app)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")
    
    @app.errorhandler(Exception)
    def handle_error(e):
        app.logger.error(f'Web app: An error occurred: {str(e)}')
        return str(e), 500

    from views import views

    app.register_blueprint(views.bp)
    return app

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
