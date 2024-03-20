import streamlit as st
import os

from src.video_object import VideoObject


def main():
    st.set_page_config(page_title="Frisbee dashboard", layout="wide")
    st.title("Tactical  dashboard")

    with st.sidebar:
        st.title("Configuration")

        st.subheader("Video uploader")
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4"])
        st.subheader("Model uploader")
        uploaded_model = st.file_uploader("Choose a model file", type=["pt"])

    if uploaded_video is not None and uploaded_model is not None:

        os.makedirs('data/videos', exist_ok=True)
        os.makedirs('data/models', exist_ok=True)

        video_path = f'data/videos/{uploaded_video.name}'
        model_path = f'data/models/{uploaded_model.name}'

        if not os.path.exists(video_path):
            with open(video_path, 'wb') as out_file:
                out_file.write(uploaded_video.read())
        
        if not os.path.exists(model_path):
            with open(model_path, 'wb') as out_file:
                out_file.write(uploaded_model.read())

        video_object = VideoObject(video_path)   

        frame_idx = st.slider("Select Frame index", 
                              min_value=0, 
                              max_value=video_object.get_num_frames() - 1 , 
                              value=50)

        frame = video_object.get_frame(frame_idx)

        col1, col2 = st.columns(2)

        col1.image(frame[... , ::-1], caption="Original Frame")
        col2.image(frame[... , ::-1], caption="Processed Frame")


if __name__ == "__main__":
    main()