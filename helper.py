from ultralytics import YOLO
import traceback
import streamlit as st
import cv2
import settings
import tempfile
import os
import winsound
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.sidebar.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def plot(conf, model, image, res, st_plot):
    try:
        for result in res:
            boxes = result.boxes.xywh
        # Extract center coordinates
        center_x = boxes[:, 0]
        center_y = boxes[:, 1]
        pil_image = Image.fromarray(image)  # Convert numpy array to PIL Image
        width, height = pil_image.size

        # Plotting
        ids = [f'{i+1}' for i in range(len(center_x))]

        fig, ax = plt.subplots(figsize=(width / 100, height / 100))  # Convert pixel to inches for figsize

        # Plot center coordinates
        ax.scatter(center_x, center_y, color='red', label='People')

        # Annotate each point with its ID
        for i, (x, y) in enumerate(zip(center_x, center_y)):
            ax.annotate(ids[i], (x, y), textcoords="offset points", xytext=(0,10), ha='center')

        # Add labels and title
        ax.set_title('Positions of Workers')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # Set plot limits to match the image dimensions
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # Show plot
        plt.legend()
        plt.grid(True)
        ax.set_aspect('equal', adjustable='box')

        st_plot.pyplot(fig)
    except Exception as e:
        # Display the error message with traceback
        st.error(f"An error occurred: {e}")
        st.text_area("Traceback", traceback.format_exc())

def _display_detected_frames(conf, model, st_frame, image, plot_person, st_plot, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """


    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)

    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
        
    #alert
    for result in res:
            if any(value in result.boxes.cls for value in [3.0, 4.0, 5.0, 6.0]):
                winsound.Beep(500, 500)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                       caption='Detected Video',
                       channels="BGR",
                       use_column_width=True
                       )
    if plot_person == 1:
        plot(conf, model, res_plotted, res, st_plot)

def play_webcam(conf, model, plot_person):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            col1, col2 = st.columns(2)
            if plot_person == 1:
                with col1:
                    st_frame = st.empty()
                with col2:
                    st_plot= st.empty()
            else:
                st_frame = st.empty()
                st_plot= st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             plot_person,
                                             st_plot,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_rtsp_stream(conf, model, plot_person):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url")
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            frame_count = 0
            col1, col2 = st.columns(2)
            if plot_person == 1:
                with col1:
                    st_frame = st.empty()
                with col2:
                    st_plot= st.empty()
            else:
                st_frame = st.empty()
                st_plot= st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                frame_count += 1
                if success:
                    if frame_count % 3 != 0:
                        continue
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             plot_person,
                                             st_plot,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    is_display_tracker, tracker = display_tracker_options()
    source_vid = st.sidebar.file_uploader(
        "Choose a video...", type=("mp4", "avi", "mov"))
    if source_vid is not None:
        st.video(source_vid)
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, source_vid.name)
        with open(path, "wb") as f:
                f.write(source_vid.getvalue())
        st.write("Uploaded Video:", source_vid.name)
    else:
        st.warning("Please upload a video file.")


    if st.sidebar.button('Detect Video Objects'):
        try:
            cap = cv2.VideoCapture(path)
            st_frame = st.empty()
            frame_count = 0
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()
                frame_count += 1
                if success:
                    if frame_count % 3 != 0:
                        continue
                    # Run YOLOv8 inference on the frame
                    if is_display_tracker:
                        results  = model.track(frame, conf=conf, persist=True, tracker=tracker)
                    else:
                        # Predict the objects in the image using the YOLOv8 model
                        results  = model.predict(frame, conf=conf)

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                    # Display the annotated frame
                    st_frame.image(annotated_frame,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Break the loop if the end of the video is reached
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
