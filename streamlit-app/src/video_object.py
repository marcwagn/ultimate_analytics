import cv2

class VideoObject():    
    def __init__(self, video_path: str) -> None:
        """
        Initializes a VideoObject instance.

        Args:
            video_path (str): The path to the video file.

        Returns:
            None
        """
        self.video_path = video_path

    def get_frame(self, frame_index: int):
        """
        Retrieves a specific frame from the video.

        Args:
            frame_index (int): The index of the frame to retrieve.

        Returns:
            numpy.ndarray or None: The retrieved frame as a NumPy array, or None if the frame retrieval failed.
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame")
            return None

        cap.release()

        return frame
    
    def get_num_frames(self):
        """
        Returns the number of frames in the video.

        Returns:
            int: The number of frames in the video.
        """
        cap = cv2.VideoCapture(self.video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return num_frames