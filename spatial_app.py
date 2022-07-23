from turtle import width
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from openslide import OpenSlide
import os
from heatmap_survival import generate_heatpmap_survival
from PIL import Image
import pandas as pd

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
#stroke_color = st.sidebar.color_picker("Box border color: ")
stroke_color = "#000"
test_mode = st.sidebar.selectbox("Test mode (only 100 patches will be predicted):", ("False", "True"))
cell_type_button = st.sidebar.button("Get cell type visualization")
prognosis_button = st.sidebar.button("Get prognosis visualization")
clear_button = st.sidebar.button("Clear the session")

device = check_device(config['use_cuda'])
with st.sidebar:
    st.write('Device available:', device)

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

max_patches_per_slide = np.inf
if test_mode == "True":
    max_patches_per_slide = 100

if cell_type_button:
    with st.spinner('Reading patches...'):
        dataloader = read_patches(slide, max_patches_per_slide)

    with st.spinner('Loading model...'):
        model = load_model(checkpoint='model_cell.pt', config = config)
     
    with st.spinner('Predicting cell types...'):
        results = predict_cell(model, dataloader, device=device)
    
    with st.spinner('Generating visualizations...'):
        heatmap, percentages = generate_heatmap(slide, patch_size= (112,112), labels=results, config=config)

    im = Image.fromarray(heatmap)
    legend = Image.open('pictures/cell_type.png')
    col1, mid, col2 = st.columns([100,1,20])
    with col1:
        st.image(im, caption='Cell distribution accross the tissue')
    with col2:
        st.image(legend)
    
    # Display statistic tables for cell proportions 
    st.markdown('<p class="big-font">Percentage of each cell type in the slide</p>', unsafe_allow_html=True)
    df = pd.DataFrame([percentages])
    df = df.T
    df.columns = ['Percentage (%)']
    df['Cell type'] = df.index
    df = df[['Cell type', "Percentage (%)"]]
    df = df.reset_index(drop = True)

    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    html_string = '''
    <html>
    <link rel="stylesheet" type="text/css" href=""/>
    <body>
        {table}
    </body>
    </html>
    '''
    table_html = df.style.set_properties(
    subset=['Cell type'], 
    **{'font-weight': 'bold', 'text-align': 'center'}).render()
    #st.table(df)
    st.write(html_string.format(table=table_html), unsafe_allow_html=True)
    #st.image([im, legend], caption=['Cell distribution accross the tissue', None])

if prognosis_button:
    with st.spinner('Reading patches...'):
        dataloader = read_patches(slide)

    config['num_classes'] = 1
    with st.spinner('Loading model...'):
        model = load_model(checkpoint='model_survival.pt', config = config)
    
    with st.spinner('Predicting survival...'):
        results = predict_survival(model, dataloader, device=device)

    config['label_column'] = 'risk_score'
    with st.spinner('Generating visualizations...'):
        heatmap = generate_heatpmap_survival(slide, patch_size= (112,112), 
                                            results=results, 
                                            compress_factor=config['compress_factor'])

    im = Image.fromarray(heatmap)
    st.image(im, caption='Survival prediction accross the tissue')

if clear_button:
    clear(path)