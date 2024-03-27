FROM python:3.11-bookworm

COPY src /app/src
COPY requirements* /app

WORKDIR /app

# Register Google Cloud SDK and Fuse repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-bookworm main" | tee /etc/apt/sources.list.d/gcsfuse.list
# Add GCS keys
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt update && \
    apt install -y \ 
    # The missing requirements for OpenCV
    libgl1 libglib2.0-0 \
    # Useful for network troubleshooting
    netcat-openbsd dnsutils \
    # Google Cloud Storage components
    google-cloud-cli fuse gcsfuse

RUN pip install -r requirements.lock

# Install extra GCP-specific packages
RUN pip install google-cloud-logging

# Install nginx and supervisord 
RUN apt-get -y install nginx supervisor openssl
RUN mkdir -p /var/log/supervisor

# Deploy NGINX conf
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/project.conf /etc/nginx/conf.d/
COPY nginx/forward_headers.conf /etc/nginx/include/forward_headers.conf
# Deploy supervisord conf
COPY supervisord/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN chmod -R a+w /var/log/supervisor

WORKDIR /app/src

RUN groupadd -r svc_user -g 433 && \
    useradd -u 431 -r -g svc_user -s /bin/bash -c "Unprivileged user" svc_user
#USER svc_user

CMD ["/usr/bin/supervisord"]