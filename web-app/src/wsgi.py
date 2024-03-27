from gevent.pywsgi import WSGIServer
import os
from app import create_app

if __name__ == "__main__":
    # A production-ready WSGI server for the Flask app
    flask_app = create_app()
    flask_app.logger.info("About to serve the web app via WSGI")
    port = int(os.environ.get('INTERNAL_PORT', 5000))
    net_interface = "0.0.0.0"
    http_server = WSGIServer((net_interface, port), flask_app)
    flask_app.logger.info(f"WSGI listening on network interface {net_interface} and port {port}")
    http_server.serve_forever()