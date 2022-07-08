import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from openslide import OpenSlide
import os
from heatmap_survival import generate_heatpmap_survival
from PIL import Image

from utils import *
from spa_mapping import generate_heatmap
from heatmap_survival import generate_heatpmap_survival
# Some easy style stolen from internet...
st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

if not os.path.exists('temp'):
    os.mkdir('temp')

# Main config
config = {
  "model_name": "resnet50",
  "target_label": "max_class",
  "num_classes": 6,
  "batch_size": 128,
  "use_cuda":True,
  "data_path": "/oak/stanford/groups/ogevaert/data/Spatial_Heiland/data/patches_spot",
  'label_column' : 'label',
  "num_workers": 20,
  "num_epochs": 10,
  "img_size": 46,
  "lr": 5e-4,
  "weight_decay": 1e-5,
  "weighted_sampler": True,
  "pretrained" : True,
  "train_bag_size":1,
  "val_bag_size":1,
  "aggregator": "identity",
  "aggregator_hdim": 2048,
  "task" : "classification",
  "n_layers_to_train":2,
  "flag": "model_pathology",
  "max_patch_per_wsi_train" :100000,
  "max_patch_per_wsi_val" : 100000,
  "restore_path": "",
  "compress_factor": 16
}

# Specify canvas parameters in application
bg_image = st.sidebar.file_uploader("Image:", type=["tiff","svs"])
# TODO max width / height
#model_type = st.sidebar.selectbox("Model:", ("ResUNet", "ResUNet2", "Combined_1_2"))
#stroke_color = st.sidebar.color_picker("Box border color: ")
stroke_color = "#000"
cell_type_button = st.sidebar.button("Get cell type visualization")
prognosis_button = st.sidebar.button("Get prognosis visualization")
clear_button = st.sidebar.button("Clear the session")

# Create a canvas component
canvas_result = None
if bg_image:
    path = save_uploaded_file(bg_image)
    slide = OpenSlide(path)
    image = slide.get_thumbnail(size=(512,512))
    st.image(image)
    bg_image = None
else:
    st.markdown('<p class="big-font">Choose an image first</p>', unsafe_allow_html=True)

# https://stackoverflow.com/questions/66372402/conversion-of-dimension-of-streamlit-canvas-image
# https://github.com/andfanilo/streamlit-drawable-canvas/issues/39


if cell_type_button:
    with st.spinner('Reading patches...'):
        dataloader = read_patches(slide)

    with st.spinner('Loading model...'):
        model = load_model(checkpoint='model_cell.pt', config = config)
     
    with st.spinner('Predicting cell types...'):
        results = predict_cell(model, dataloader)
    
    with st.spinner('Generating visualizations...'):
        heatmap = generate_heatmap(slide, patch_size= (112,112), labels=results, config=config)

    im = Image.fromarray(heatmap)
    st.image(im, caption='Cell distribution accross the tissue')

if prognosis_button:
    with st.spinner('Reading patches...'):
        dataloader = read_patches(slide)

    config['num_classes'] = 1
    with st.spinner('Loading model...'):
        model = load_model(checkpoint='model_survival.pt', config = config)
    
    with st.spinner('Predicting survival...'):
        results = predict_survival(model, dataloader)

    config['label_column'] = 'risk_score'
    with st.spinner('Generating visualizations...'):
        heatmap = generate_heatpmap_survival(slide, patch_size= (112,112), results=results)

    im = Image.fromarray(heatmap)
    st.image(im, caption='Survival prediction accross the tissue')

if clear_button:
    clear(path)