import streamlit as st
import requests
import cv2
from PIL import Image
import io

API_URL = "http://192.168.0.102:8000/"





###################################################################
######## Main Work ################################################
###################################################################
def main():
    st.title("Object Detection with FastAPI")

    # Choose the input source
    input_option = st.radio("Select Input Source:", ("Image", "Webcam", "Video"))

    if input_option == "Image":
        image_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Objects"):
                detect_objects(image)

    elif input_option == "Webcam":
        st.write("Webcam not supported in Streamlit. Please use the Video option instead.")

    else:  # Video
        video_file = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov"])
        if video_file is not None:
            video_bytes = video_file.read()
            st.video(video_bytes)

            if st.button("Detect Objects"):
                detect_objects_from_video(video_file)
                
                
###################################################################
######## Helper Functions -detect obj from image ##################
###################################################################



def detect_objects(image):
    API_IMG = "http://localhost:8000/predict_to_image" # hardcoded

    # Convert the PIL image to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="JPEG")
    img_byte_array = img_byte_array.getvalue()

    response = requests.post(API_IMG, files={"file": img_byte_array})
    if response.status_code == 200:
        st.image(Image.open(io.BytesIO(response.content)), caption="Detected Image", use_column_width=True)
    else:
        st.error("Error detecting objects.")
        
        
        
###################################################################
######## Helper Functions -detect obj from video ##################
###################################################################


def detect_objects_from_video(video_bytes):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_bytes)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform object detection on the frame
        detect_objects(image)

        # Display the processed frame
        st.image(image, use_column_width=True)

        # Uncomment the following line to enable real-time video processing
        # time.sleep(0.1)

    cap.release()

    # st.video(video_bytes)


if __name__ == "__main__":
    main()
