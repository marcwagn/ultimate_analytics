FROM python:3.11.8-slim-bullseye

COPY src /app/src
COPY requirements* /app

WORKDIR /app

RUN apt update && \
    # The missing requirements for OpenCV
    apt install -y libgl1 libglib2.0-0 && \
    # Useful for network troubleshooting
    apt install -y netcat dnsutils

RUN pip install -r requirements.lock

WORKDIR /app/src

EXPOSE 5000:5000
CMD python app.py