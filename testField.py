from PIL import Image
from ultralytics import YOLO
import streamlit as st
import pandas as pd
import cv2
st.set_page_config(layout="wide")
tab1,tab2,tab3,tab4=st.tabs(['Demo','Model Card','Live Feed','About'])
def string_to_json(input_string):
    # Split the input string by comma and remove any leading or trailing whitespace
    items = [item.strip() for item in input_string.split(',')][:-1]
    results=[]
    for item in items:
        number,label=item.split(' ')
        number=int(number)
        label=str(label)
        dict1 = {"class":label,"Quantity":number}
        results.append(dict1) 
    try:  
        return pd.DataFrame(results)
    except:
        return results

def process_YOLO_results(results):
    for r in results:
        im_array = r.plot(conf=True)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        try:
            ssummary = string_to_json(r.verbose()) # save to file
            #st.write(r.tojson())
        except:
            ssummary = r.verbose()
    return im,ssummary
    
with tab1:
    model_choice=st.sidebar.selectbox('Select a model',('yolov5su','yolov8m','best'),index=1)
    model = YOLO(f'/home/stelios/Desktop/Thesis/models/{model_choice}.pt',verbose=True)
    uploaded_img = st.sidebar.file_uploader("Choose an image...", type="jpg")
    if uploaded_img is not None:
        with st.spinner('Processings...'):
            confidence=st.sidebar.slider('Confidence',0.0,1.0,0.5,0.01)
            results = model(f'imgs/{uploaded_img.name}',conf=confidence)  # results list

        c1,c2 = st.columns(2)
        with c1:
            st.header('Original Image')
            st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)
        with c2:
            st.header('Processed Image')
            im,summary = process_YOLO_results(results)
            st.image(im, caption='Image', use_column_width=True)
        st.header('Result Overview')
        st.write(summary,use_column_width=True)
    else :
        st.warning('Please upload an image')

with tab2:
    st.markdown('''# Model Card: YOLOv8m Object Detection Model

## Model Details

- **Model Name:** YOLOv8m
- **Model Version:** 1.0
- **Model Type:** Object Detection
- **Framework:** PyTorch

## Intended Use 

- **Primary Purpose:** Real-time object detection of Electronic Components.
- **Intended Users:** Developers, researchers, and organizations deploying object detection on mobile platforms.
- **Intended Environment:** Shorting E-Waste Station.

## Evaluation

- **Evaluation Metrics:** Mean Average Precision (mAP).
- **Performance:** High Performance on the test set.
- **Limitations:** Performance may degrade under challenging conditions (e.g., occlusion, extreme lighting).

## Acknowledgments

Acknowledgments to the open-source community and researchers.
''')
    with st.expander('Experiment 1'):
        st.markdown('''
    ## Experiment 1: Object Detection on Electronic Components
    - **Objective:** Evaluate the model's performance on electronic components.
    - **Data:** 
      - A dataset of electronic components
      - **Size:** 1000 images
      - **Distribution:** 10 classes
      - **Evaluation Metric:** Mean Average Precision (mAP).
    - **Results:** The model achieved high precision and recall on the test set.
    - **Conclusion:** The model is suitable for detecting electronic components in real-time.

                    ''')
    with st.expander('Experiment 2'):
        st.markdown('''''')
    st.write(model_choice)

with tab3:
    st.title("Webcam Live Feed")
    
    # Checkbox to start/stop the webcam feed
    run = st.checkbox('Run')
    
    # Placeholder for displaying the webcam feed
    frame_placeholder = st.empty()
    
    # Start capturing video from the webcam
    camera = cv2.VideoCapture(0)
    
    # Continuously read and display frames from the webcam
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame from webcam. Please try again.")
            break
        
        # Convert the frame from BGR to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_frame,sam=process_YOLO_results(model(frame,stream=True))
        # Display the frame in the Streamlit app
        frame_placeholder.image(yolo_frame, channels='RGB')
    
    # Release the webcam and close the Streamlit app when done
    camera.release()
    st.write("Webcam feed stopped.")

with tab4:
    st.write('About')