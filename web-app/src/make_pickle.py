import pickle
import sys
from tasks.precalculated import _calculate_sha256sum
from tasks.tasks import video_analysis
from pathlib import Path
import os

def main():
    """
    Command line tool to run Ultimate board detection and pickle the results.
    The result data is a Pythondictionary in the following format:
        key (str): frame number
        value (object):
            cls (int): class id of the detected object (0: player; 29: frisbee disc; 30: referee)
            x (float): mid-point x coordinate of the detected object, expressed in 'real ultimate pitch' units
            y (float): mid-point y coordinate of the detected object, expressed in 'real ultimate pitch' units
            team (int): team number (typically 0 or 1)
            id (int): YOLO instance id of the detected object; can be used to keep track of objects across multiple frames
    """
    if len(sys.argv) < 2:
        print(f"{sys.argv[0]} is a tool to run Ultimate board detection and pickle the results for a given Ultimate Frisbee video.")
        print("The results are saved to a file <video_sha256>.pickle in [output_dir].")
        print(f"Usage: {sys.argv[0]} path_to_video [output_dir]")
        sys.exit(-1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else "./data/precalculated"
    print(f"Using output_dir: {output_dir}")
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        print(f"Error: {output_dir} needs to be an existing directory")
        sys.exit(-1)

    video_sha_sum = _calculate_sha256sum(video_path)
    print(f"SHA-256 checksum of {video_path} is {video_sha_sum}")

    analysis_results = video_analysis(video_path)
    if analysis_results is not None and "coordinates" in analysis_results:
        streamlined_results = analysis_results["coordinates"]
    else:
        raise RuntimeError("Unexpected results returned from Ultimate Board video_analysis")
    
    output_file = Path(output_dir)/f"{video_sha_sum}.pickle"
    with open(output_file, "wb") as f:
            pickle.dump(streamlined_results, f)
    print(f"Saved results as {output_file}")

if __name__ == "__main__":
    main()