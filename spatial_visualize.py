import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from openslide import OpenSlide
import os
from heatmap_survival import generate_heatpmap_survival
import os
from glob import glob
import base64

from utils import *
from spa_mapping import generate_heatmap
from heatmap_survival import generate_heatpmap_survival
# Some easy style stolen from internet...

st.set_page_config(layout="wide")
st.markdown("""
<style>
.container {
        display: flex;
    }
.big-font {
    font-size:30px !important;
}
.cell-img {
        float:right;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title('Cell types and prognosis visualization')
def file_selector(folder_path='.'):
    filenames = glob(os.path.join('spatial_vis_TCGA', folder_path+'*-heatmap.png'))
    new_filenames = [x.replace('-heatmap.png', '') for x in filenames]
    selected_filename = st.sidebar.selectbox('Select a patient file', new_filenames)
    return selected_filename

user_input = st.sidebar.text_input("Paste the patient ID")
src = None
if len(user_input) != 0:
    src = file_selector(folder_path=user_input)
    if src:
        im_heatmap = Image.open(src+'-heatmap.png')
        im_histo = Image.open(src+'-histo.png')
        
        st.markdown('<p class="big-font">Original slide</p>', unsafe_allow_html=True)
        st.image(im_histo, caption='Original slide')
        st.markdown('<p class="big-font">Cell types prediction</p>', unsafe_allow_html=True)
        col1, mid, col2 = st.columns([100,1,20])
        with col1:
            st.image(im_heatmap, caption='Cell distribution accross the tissue')
        with col2:
            cell_label = Image.open('pictures/cell_type.png')
            st.image(cell_label)

else:
    st.markdown('<p class="big-font">Please, introduce a Patient ID from TCGA</p>', unsafe_allow_html=True)
