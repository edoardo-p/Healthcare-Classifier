import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import SimpleITK as stk
from streamlit_cropper import st_cropper
from PIL import Image
import pickle 
import base64  #to load a saved model #to open .gif files in streamlit app

if __name__ == "__main__":
    st.cache(suppress_st_warning=True)
    st.set_page_config(layout="wide")
    app_mode = st.sidebar.radio('Select Page',['Home Page','Prediction','privacy policy']) 
    uploaded_data=None

    if app_mode=='Home Page': 
        st.title("Radiogram [o]*") 
        uploaded_data = st.file_uploader("Upload the CT scan", accept_multiple_files=False) 
        st.set_option('deprecation.showfileUploaderEncoding', False)

        # Upload an image and set some options for demo purposes
        st.header("Region Of Interest")
        img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg', 'npy'])
        realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
        box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
        aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "2:3": (2, 3),
            "Free": None
        }
        aspect_ratio = aspect_dict[aspect_choice]

        if img_file:
            ct_scan = np.load(img_file)
            slc = 0
            if not realtime_update:
                st.write("Double click to save crop")
            # Get a cropped image from the frontend
            cropped_img1 = st_cropper(Image.fromarray(ct_scan[slc, :, :]), realtime_update=realtime_update, box_color=box_color,
                                        aspect_ratio=aspect_ratio)
            #if st.button('<-'):
                 
                 
            #if st.button('->'):
            
            # Manipulate cropped image at will
            cropped_img1 =Image.open(cropped_img1)
            st.write("Preview")
            _ = cropped_img1.thumbnail((150,150))
            st.image(cropped_img1)
        
        if uploaded_data is not None:
             img_ct =stk.ImageTead (uploaded_data)


    if app_mode =='privacy policy':
            st.title('Data and Security')
            st.text("Personal data shall be:(a) processed lawfully, fairly and in a transparent manner in relation to individuals  (lawfulness, fairness and transparency)")
            st.text("(b) collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes; further processing for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes shall not be considered to be incompatible with the initial purposes (purpose limitation)")
            st.text("(c) adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed (data minimisation)")
            st.text("(d) accurate and, where necessary, kept up to date; every reasonable step must be taken to ensure that personal data that are inaccurate, having regard to the purposes for which they are processed, are erased or rectified without delay (accuracy)")
            st.text("(e) kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed; personal data may be stored for longer periods insofar as the personal data will be processed solely for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes subject to implementation of the appropriate technical and organisational measures required by the GDPR in order to safeguard the rights and freedoms of individuals (storage limitation)")
            st.text("(f) processed in a manner that ensures appropriate security of the personal data, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage, using appropriate technical or organisational measures (integrity and confidentiality).")
            st.checkbox('I agree')
            st.download_button('confirm',img_ct)

    if app_mode == 'Prediction':  
            st.title('Result of analysis')

    















