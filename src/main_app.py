"""
Landing point of GBM360
"""

import streamlit as st
import os
from utils import *
import run_predict
import auth
import about
import re
import base64
import pdb

st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
.streamlit-expanderHeader {
    font-size: x-large;
    font-weight: bold;
}
button[data-baseweb="tab"] {
  font-size: 26px;
  font-weight: bold;

}
</style>
""", unsafe_allow_html=True)

st.image("pictures/logo.png", width = 150)

tab1, tab2, tab3 = st.tabs(["About", "Tutorial", "Run"])

with tab1:

    about.app()

with tab2:

    def markdown_images(markdown):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
        images = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
        #images = re.findall(r'<img\s+src="([^"]+)"\s+width="([^"]+)"\s+height="([^"]+)"\s*>', markdown)
        return images

    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded

    def img_to_html(img_path, img_alt):
        img_format = img_path.split(".")[-1]
        img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 70%;">'

        return img_html

    def markdown_insert_images(markdown):

        images = markdown_images(markdown)
        for image in images:
            image_markdown = image[0]
            image_alt = image[1]
            image_path = image[2]

            if os.path.exists(image_path):
                markdown = markdown.replace(image_markdown, img_to_html(image_path, image_alt))
        return markdown

    with open("src/tutorial.md", "r") as readme_file:
        readme = readme_file.read()

    readme = markdown_insert_images(readme)
    
    with st.container():
        st.markdown(readme, unsafe_allow_html=True)
        
with tab3:
    if not os.path.exists('temp'):
        os.mkdir('temp')

    run_predict.app()
        

# if not st.session_state["authentication_status"]:
#     st.markdown('<p class="big-font">Please log in first.</p>', unsafe_allow_html=True)
# else: