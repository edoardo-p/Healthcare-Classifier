import diagnose
import numpy as np
import pages
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

# from streamlit_modal import Modal


def main():
    st.set_page_config(layout="centered")

    if "page" not in st.session_state:
        st.session_state["page"] = "upload"

    if st.session_state["page"] == "login":
        pages.login_page()

    # app_mode = st.sidebar.radio(
    #     "Select Page", ["Home Page", "Prediction", "privacy policy"]
    # )

    if st.session_state["page"] == "upload":
        pages.upload_page()

    if st.session_state["page"] == "segmentation":
        pages.segmentation_page()

    if st.session_state["page"] == "prediction":
        pages.prediction_page()


if __name__ == "__main__":
    slc = 0
    main()
