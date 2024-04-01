FROM python:3.11-bookworm

COPY src /app/src
COPY scripts /app/scripts
COPY requirements* /app

WORKDIR /app

RUN apt update && \
    apt install -y \ 
    # The missing requirements for OpenCV
    libgl1 libglib2.0-0 \
    # Useful for network troubleshooting
    netcat-openbsd dnsutils

RUN pip install -r requirements.lock

# Special case Pytorch and Ultralytics to force the CPU-only version
RUN pip install --no-cache torch torchvision ultralytics --extra-index-url https://download.pytorch.org/whl/cpu

# Install extra GCP-specific packages
RUN chmod +x scripts/*.sh && scripts/install-gcs.sh

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

#RUN chown -R svc_user.svc_user /app

#USER svc_user

CMD ["/usr/bin/supervisord"]