import numpy as np
import SimpleITK as sitk
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from bounding_box import BoundingBox

bb = BoundingBox(64)


def main():
    # st.cache(suppress_st_warning=True)
    st.set_page_config(layout="wide")
    app_mode = st.sidebar.radio(
        "Select Page", ["Home Page", "Prediction", "privacy policy"]
    )
    uploaded_data = None

    if app_mode == "Home Page":
        st.title("Radiogram")
        img_file = st.file_uploader(
            "Upload the CT scan",
            type=["png", "jpg", "npy", "nii"],
            accept_multiple_files=False,
        )
        st.set_option("deprecation.showfileUploaderEncoding", False)

        # Upload an image and set some options for demo purposes
        st.header("Region of interest")
        box_color = st.color_picker(label="Box Color", value="#0000FF")

        if img_file:
            ct_scan = np.load(img_file)
            if "slice" not in st.session_state:
                st.session_state["slice"] = 159
            slc = st.session_state["slice"]
            # ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
            # Get a cropped image from the frontend
            cropped = st_cropper(
                Image.fromarray(ct_scan[slc, :, :]),
                box_color=box_color,
                aspect_ratio=(1, 1),
                box_algorithm=bb.get_box,
            )
            st.write(st.session_state["slice"])
            if st.button("<-"):
                st.session_state["slice"] = slc - 1

            if st.button("->"):
                st.session_state["slice"] = slc + 1

            # Manipulate cropped image at will
            # cropped_img1 = Image.open(cropped_img1)
            # st.write("Preview")
            # cropped.thumbnail((150, 150))
            # st.image(cropped)

    if app_mode == "privacy policy":
        st.title("Data and Security")
        st.write("Personal data shall be:")
        st.write(
            "(a) processed lawfully, fairly and in a transparent manner in relation to individuals  (lawfulness, fairness and transparency)"
        )
        st.write(
            "(b) collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes; further processing for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes shall not be considered to be incompatible with the initial purposes (purpose limitation)"
        )
        st.write(
            "(c) adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed (data minimisation)"
        )
        st.write(
            "(d) accurate and, where necessary, kept up to date; every reasonable step must be taken to ensure that personal data that are inaccurate, having regard to the purposes for which they are processed, are erased or rectified without delay (accuracy)"
        )
        st.write(
            "(e) kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed; personal data may be stored for longer periods insofar as the personal data will be processed solely for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes subject to implementation of the appropriate technical and organisational measures required by the GDPR in order to safeguard the rights and freedoms of individuals (storage limitation)"
        )
        st.write(
            "(f) processed in a manner that ensures appropriate security of the personal data, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage, using appropriate technical or organisational measures (integrity and confidentiality)."
        )
        st.checkbox("I agree")
        # st.download_button("confirm", img_ct)

    if app_mode == "Prediction":
        st.title("Result of analysis")


if __name__ == "__main__":
    slc = 0
    main()
