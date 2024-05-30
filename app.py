# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="PPE Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Automatic surveillance system for PPE compliance in construction sites")

# Sidebar
st.sidebar.header("ML Model Config")

model_type = st.sidebar.radio(
    "Select Model", ['PPE', 'Worker'])

# Model type
if model_type == 'PPE':
    model_path = Path(settings.PPE_MODEL)
    plot_person = 0
elif model_type == 'Worker':
    model_path = Path(settings.WORKER_MODEL)
    plot_person = 1


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 0, 100, 50)) / 100

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)
    if source_img is None:
        default_image_path = str(settings.DEFAULT_IMAGE)
        default_image = PIL.Image.open(default_image_path)
        st.image(default_image_path, caption="Default Image",
                             use_column_width=True)
    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)

            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)

        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model, plot_person)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model, plot_person)

else:
    st.error("Please select a valid source type!")
