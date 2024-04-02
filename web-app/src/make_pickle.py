import pickle
import sys
from tasks.precalculated import _calculate_sha256sum
from tasks.tasks import video_analysis
from pathlib import Path
import os

# def _calculate_sha256sum(filename):
#     with open(filename, 'rb', buffering=0) as f:
#         return hashlib.file_digest(f, 'sha256').hexdigest()
    
# Command line tool to run Ultralytics YOLO object tracking and pickle the results

# def _track():
#     model = ultralytics.YOLO(model_path, verbose=True)
#     yolo_progress_reporting_event = "on_predict_batch_start"
#     progress_callback_wrapped = make_callback_adapter_with_counter(yolo_progress_reporting_event, 
#                                                                     lambda _,counter: progressbar_callback(counter))
#     model.add_callback(yolo_progress_reporting_event, progress_callback_wrapped)

#     # NB - if torch package is installed in the CPU variant, the device will default to "cpu"
#     device = 0 if torch.cuda.is_available() else "cpu" 
#     tracking_results = model.track(source=video_path, agnostic_nms=True, show=False, device=device, stream=True)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} path_to_video [output_dir]")
        sys.exit(-1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else "./src/data/precalculated"
    print(f"Using output_dir: {output_dir}")
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        print(f"Error: {output_dir} needs to be an existing directory")
        sys.exit(-1)

    video_sha_sum = _calculate_sha256sum(video_path)
    print(f"SHA-256 checksum of {video_path} is {video_sha_sum}")
    #model_path = "./data/model/best.pt"

    analysis_results = video_analysis(video_path)
    if analysis_results is not None and "coordinates" in analysis_results:
        streamlined_results = analysis_results["coordinates"]
    else:
        raise RuntimeError("Unexpected results returned from video_analysis")
    # tracking_results = _track(model_path=model_path, video_path=video_path, progressbar_callback=None)

    # print("Streamlining the results...")
    # streamlined_results = []

    # for i, result in enumerate(tracking_results):
    #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    #     jpg_result, encimg = cv2.imencode('.jpg', result.orig_img, encode_param)
    #     if not jpg_result:
    #          raise RuntimeError(f"JPEG compression failed for frame {i}")
    #     result.orig_img = encimg
    #     streamlined_results.append(result)

    # Removing orig_img
    # for result in tracking_results:
    #     del result.orig_img
    #     streamlined_results.append(result)

    output_file = Path(output_dir)/f"{video_sha_sum}.pickle"
    with open(output_file, "wb") as f:
            pickle.dump(streamlined_results, f)
    print(f"Saved results as {output_file}")

if __name__ == "__main__":
    main()