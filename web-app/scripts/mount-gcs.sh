# Mount the Google Cloud Storage bucket
# Variables:
#   GCS_BUCKET - required
#   MOUNT_DIR - optional

if [ -z ${GCS_BUCKET+x} ]; then 
    echo "Fatal: GCS_BUCKET is unset"
    exit -1
else echo "GCS_BUCKET is set to $GCS_BUCKET"
fi
MOUNT_DIR=${MOUNT_DIR:=/app/gcs}

gcsfuse --version
echo "About to mount $GCS_BUCKET at $MOUNT_DIR"
if [ ! -d $MOUNT_DIR ]; then 
    echo "Creating directory $MOUNT_DIR"
    mkdir $MOUNT_DIR
fi
chmod a+wx $MOUNT_DIR
gcsfuse --debug_gcs --debug_http --debug_fuse --debug_invariants -o ro $GCS_BUCKET $MOUNT_DIR
echo "gcsfuse returned $?"

if [ ! -d $MOUNT_DIR/precalculated ]; then
    echo "Error: expected $MOUNT_DIR/precalculated to exist in the mounted bucket, but nothing found"
    exit -2
fi