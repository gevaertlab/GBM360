"""
The run prediction page
"""

import numpy as np
from PIL import Image
import streamlit as st
from openslide import OpenSlide
from utils import *
from spa_mapping import generate_heatmap_cell_type, generate_heatpmap_survival
from spatial_stat import gen_graph, compute_percent
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import json

def app():

    with open("config/config.json", 'r') as f:
        config = json.load(f)

    # Specify canvas parameters in application
    bg_image = st.file_uploader("Image:", type=["tiff","svs"])

    # Control panel
    example_button = st.button("Use an example slide")
    test_mode = st.selectbox("Test mode (only 1000 patches will be predicted):", ("True", "False"))
    cell_type_button = st.button("Get cell type visualization")
    prognosis_button = st.button("Get prognosis visualization")
    clear_button = st.button("Clear the session")

    # Check available device
    device = check_device(config['use_cuda'])
    st.write('Device available:', device)

    # Initialization
    if 'slide' not in st.session_state:
        st.session_state.slide = None
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'dataloader' not in st.session_state:
        st.session_state.dataloader = None

    if bg_image:
        path = save_uploaded_file(bg_image)
        st.session_state.slide = OpenSlide(path)
        st.session_state.image = st.session_state.slide.get_thumbnail(size=(512,512))
        st.image(st.session_state.image)
        bg_image = None

    if example_button:
        path = os.path.join("example", "C3L-00016-21.svs")
        st.session_state.slide = OpenSlide(path)
        st.session_state.image = st.session_state.slide.get_thumbnail(size=(512,512))
        st.image(st.session_state.image)
    
    max_patches_per_slide = np.inf
    if test_mode == "True":
        max_patches_per_slide = 1000

    if cell_type_button and st.session_state.slide:
        slide = st.session_state.slide
        with st.spinner('Reading patches...'):
            dataloader = read_patches(slide, max_patches_per_slide)

        with st.spinner('Loading model...'):
            model = load_model(checkpoint='model_weights/model_cell.pt', config = config)
        
        with st.spinner('Predicting cell types...'):
            results = predict_cell(model, dataloader, device=device)
        
        with st.spinner('Generating visualization...'):
            heatmap = generate_heatmap_cell_type(slide, patch_size= (112,112), labels=results, config=config)
        im = Image.fromarray(heatmap)
        legend = Image.open('pictures/cell-type-hor.png')
        st.image(legend)
        st.image(im, caption='Cell distribution accross the tissue')

        with st.spinner('Calculating spatial statistics...'):
            df_percent = compute_percent(results) # cell type composition
            dgr_centr, im_mtx, df_cluster = gen_graph(slide, results = results) # graph statistics
        

        # Display statistic tables for cell proportions 
        color_ids, cluster_colors = get_color_ids()
        st.markdown('<p class="big-font">Cell fraction (%)</p>', unsafe_allow_html=True)
        data_container = st.container()
        with data_container:
            table, plot, _ , _ = st.columns(4)
            with table:
                st.table(data=style_table(df_percent))
            with plot:
                buf = BytesIO()
                fig, ax = plt.subplots()
                sns.barplot(data = df_percent, y = 'cell_type', x = "Percentage", palette = cluster_colors, ax = ax)
                ax.tick_params(labelsize=16)
                ax.set_ylabel('', fontdict= {'fontsize': 16, 'fontweight':'bold'})
                ax.set_xlabel('Percentage (%)',fontdict= { 'fontsize': 16, 'fontweight':'bold'})
                fig.savefig(buf, format="png", bbox_inches = "tight")
                st.image(buf)

        # Display interaction matrix
        st.markdown('<p class="big-font">Interaction matrix</p>', unsafe_allow_html=True)
        data_container = st.container()
        with data_container:
            table, plot, _ , _ = st.columns(4)
            with table:
                st.table(data=style_table(im_mtx))
            with plot:
                buf = BytesIO()
                fig, ax = plt.subplots()
                sns.heatmap(im_mtx, ax = ax)
                ax.tick_params(labelsize=16)
                fig.savefig(buf, format="png", bbox_inches = "tight")
                st.image(buf)
        
        # Display statistic tables for degree centrality
        st.markdown('<p class="big-font">Degree centrality</p>', unsafe_allow_html=True)
        data_container = st.container()
        with data_container:
            table, plot, _ , _ = st.columns(4)
            with table:
                st.table(data=style_table(dgr_centr))
            with plot:
                buf = BytesIO()
                fig, ax = plt.subplots()
                sns.barplot(data = dgr_centr, y = 'cell_type', x = 'centrality' , palette = cluster_colors, ax = ax)
                ax.tick_params(labelsize=16)
                ax.set_ylabel('', fontdict= {'fontsize': 16, 'fontweight':'bold'})
                ax.set_xlabel('Centrality score',fontdict= { 'fontsize': 16, 'fontweight':'bold'})
                fig.savefig(buf, format="png", bbox_inches = "tight")
                st.image(buf)
        
        # Display statistic tables for clustering coefficient
        st.markdown('<p class="big-font">Clustering coefficient</p>', unsafe_allow_html=True)
        data_container = st.container()
        with data_container:
            table, plot, _ , _ = st.columns(4)
            with table:
                st.table(data=style_table(df_cluster))
            with plot:
                buf = BytesIO()
                fig, ax = plt.subplots()
                sns.barplot(data = df_cluster, y = 'cell_type', x = 'cluster_coeff' , palette = cluster_colors, ax = ax)
                ax.tick_params(labelsize=16)
                ax.set_ylabel('', fontdict= {'fontsize': 16, 'fontweight':'bold'})
                ax.set_xlabel('Clustering coefficient',fontdict= { 'fontsize': 16, 'fontweight':'bold'})
                fig.savefig(buf, format="png", bbox_inches = "tight")
                st.image(buf)
    
    if prognosis_button and st.session_state.slide:

        slide = st.session_state.slide

        with st.spinner('Reading patches...'):
            dataloader = read_patches(slide, max_patches_per_slide)

        config['num_classes'] = 1
        with st.spinner('Loading model...'):
            model = load_model(checkpoint='model_weights/model_survival.pt', config = config)
        
        with st.spinner('Predicting survival...'):
            results = predict_survival(model, dataloader, device=device)
        config['label_column'] = 'risk_score'
      

        with st.spinner('Generating visualization...'):
            heatmap = generate_heatpmap_survival(slide, patch_size= (112,112), 
                                                results=results, 
                                                config = config)

        legend = Image.open('pictures/risk_score_legend.png')
        st.image(legend)
        im = Image.fromarray(heatmap)
        st.image(im, caption='Risk score prediction')

    if clear_button:
        clear(path)