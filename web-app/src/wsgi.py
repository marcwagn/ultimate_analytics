from gevent.pywsgi import WSGIServer
import os
from app import create_app

if __name__ == "__main__":
    # A production-ready WSGI server for the Flask app
    flask_app = create_app()
    port = int(os.environ.get('PORT', 5000))
    http_server = WSGIServer(("0.0.0.0", port), flask_app)
    http_server.serve_forever()