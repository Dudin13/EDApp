import streamlit as st
import sys
import os
import streamlit_drawable_canvas
import streamlit.elements.image as st_image

st.title("Diagnostic Tool")

st.write("### Environment Info")
st.write(f"Python Executable: {sys.executable}")
st.write(f"Streamlit Version: {st.__version__}")
st.write(f"CWD: {os.getcwd()}")

st.write("### Library Info")
st.write(f"streamlit-drawable-canvas path: {streamlit_drawable_canvas.__file__}")
st.write(f"Has image_to_url in st_image? {hasattr(st_image, 'image_to_url')}")

try:
    import streamlit.elements.lib.image_utils as image_utils
    st.write(f"image_utils path: {image_utils.__file__}")
    st.write(f"Has image_to_url in image_utils? {hasattr(image_utils, 'image_to_url')}")
except Exception as e:
    st.error(f"Failed to import image_utils: {e}")

st.write("### sys.path")
st.write(sys.path)
