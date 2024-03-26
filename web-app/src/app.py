from celery import Celery, Task
from flask import Flask
from flask import render_template

from flask_cors import CORS
from dotenv import load_dotenv
import os

def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__)
    CORS(app)
    app.config.from_mapping(
        CELERY=dict(
            broker_url=os.getenv("REDIS_URL", "redis://localhost"),
            result_backend=os.getenv("REDIS_URL", "redis://localhost"),
            task_ignore_result=True,
        ),
    )
    app.config.from_prefixed_env()
    celery_init_app(app)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

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
