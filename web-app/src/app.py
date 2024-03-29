from werkzeug.exceptions import HTTPException
from celery import Celery, Task
from flask import Flask, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from google.cloud import logging as gcp_logging


def create_app() -> Flask:
    load_dotenv()
    
    app = Flask(__name__)

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
    _configure_logging(app)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")
    
    @app.errorhandler(Exception)
    def handle_error(e):
        # pass through HTTP errors
        if isinstance(e, HTTPException):
            return e
        # Handling non-HTTP exceptions only
        exception_url = request.url
        app.logger.error(f'Web app: An error occurred for request URL: {exception_url}: {str(e)}')
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


def _configure_logging(app):
    app.logger.setLevel(logging.INFO)

    if os.getenv("GCP_LOGGING") == "True":
        gcp_logging_client = gcp_logging.Client()
        gcp_logging_client.setup_logging()
    else:
         # Create a console handler
        console_handler = logging.StreamHandler()
        # Create a formatter
        #formatter = logging.Formatter('%(host)s -  %(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Add the formatter to the console handler
        console_handler.setFormatter(formatter)
        # Add the console handler to the logger
        app.logger.addHandler(console_handler)


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get('INTERNAL_PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
