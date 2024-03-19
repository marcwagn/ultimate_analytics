import streamlit as st
import cv2
import tempfile
import os
from PIL import Image

def main():
    st.set_page_config(page_title="Frisbee dashboard", layout="wide")
    st.title("Tactical  dashboard")

    with st.sidebar:
        st.title("Configuration")
        st.subheader("Video uploader")

        uploaded_video = st.file_uploader("Choose a video file", type=["mp4"])
        track_bar = st.sidebar.progress(0.0,text="Progressbar for tracking")
    
    frame_count = 0

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_video.read())
            vf = cv2.VideoCapture(temp_file.name)

            tmp_folder = tempfile.mkdtemp()

            while frame_count < 100:
                ret, frame = vf.read()

                if not ret:
                    break
                
                frame_path = os.path.join(tmp_folder, f"frame_{frame_count}.jpg")
                print(frame_path)
                cv2.imwrite(frame_path, frame) 
                frame_count += 1

                track_bar.progress(frame_count/100.0)

            vf.release()

    if frame_count == 100:
        st.write("Video uploaded successfully")
        frame_num = st.slider('Which frame should be shown', 0, 100, 1)
        frame_path = os.path.join(tmp_folder, f"frame_{frame_num}.jpg")
        st.image(frame_path, caption="Uploaded video", use_column_width=True)
        
        




if __name__ == "__main__":
    main()