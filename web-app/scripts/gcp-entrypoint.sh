#!/usr/bin/env bash

# Entrypoint for the app hosted on Google Cloud Platform

# Mount the Google Cloud Storage bucket
gcsfuse --version
MOUNT_DIR=/gcs
echo "About to mount $GCS_BUCKET at $MOUNT_DIR"
if [ ! -d $MOUNT_DIR ]; then mkdir $MOUNT_DIR; fi
gcsfuse -o ro $GCS_BUCKET $MOUNT_DIR
echo "gcsfuse returned $?"
ls -l $MOUNT_DIR

# Run supervisord
/usr/bin/supervisord