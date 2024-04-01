#!/usr/bin/env bash

# Entrypoint for the app hosted on Google Cloud Platform

# Mount the Google Cloud Storage bucket
$(dirname "$0")/mount-gcs.sh

# Run supervisord
/usr/bin/supervisord