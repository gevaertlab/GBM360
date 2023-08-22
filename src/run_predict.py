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
import pyvips

def app():

    with open("config/config.json", 'r') as f:
        config = json.load(f)

    # Specify canvas parameters in application
    bg_image = st.file_uploader("Image:", type=["tiff", 'tif', "svs"])

    # Control panel
    example_button = st.button("Use an example slide")
    test_mode = st.selectbox("Run Mode:", ("Test mode (only 1,000 patches will be predicted)", "Complete"))
    st.markdown("**Note**: We are currently working on obtaining GPU support for this software. To expedite the process, the default mode "
                "is now set to `Test mode`, which will only predict 1,000 patches of the image. "
                "To predict the entire image, please switch to the `Complete` mode.")

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

        if path.endswith("tiff") or path.endswith("tif"):
            image = pyvips.Image.new_from_file(path)
            image.write_to_file("temp/test.tiff", pyramid=True, tile=True)
            path = "temp/test.tiff"

        st.session_state.slide = OpenSlide(path)
        st.session_state.image = st.session_state.slide.get_thumbnail(size=(512,512))
        st.image(st.session_state.image)
        bg_image = None

    if example_button:
        path = os.path.join("example", "C3L-00365-21.svs")
        st.session_state.slide = OpenSlide(path)
        st.session_state.image = st.session_state.slide.get_thumbnail(size=(512,512))
        st.image(st.session_state.image)
    
    max_patches_per_slide = np.inf
    if test_mode == "Test mode (only 1,000 patches will be predicted)":
        max_patches_per_slide = 1000

    if cell_type_button and st.session_state.slide:
        slide = st.session_state.slide
        with st.spinner('Reading patches...'):
            dataloader = read_patches(slide, max_patches_per_slide)

        with st.spinner('Loading model...'):
            model = load_model(checkpoint='model_weights/train_2023-04-28_prob_multi_label_weighted/model_cell.pt', config = config)
        
        with st.spinner('Predicting transcriptional subtypes...'):
            results = predict_cell(model, dataloader, device=device)
        
        with st.spinner('Generating visualization...'):
            heatmap = generate_heatmap_cell_type(slide, patch_size= (112,112), labels=results, config=config)
        im = Image.fromarray(heatmap)
        legend = Image.open('pictures/cell-type-hor.png')
        st.image(legend)
        st.image(im, caption='Subtype distribution across the tissue')

        with st.spinner('Calculating spatial statistics...'):
            df_percent = compute_percent(results) # cell type composition
            dgr_centr, im_mtx_slide, im_mtx_row, df_cluster = gen_graph(slide, results = results) # graph statistics
        
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
                sns.barplot(data = df_percent, y = 'Subtype', x = "Percentage", palette = cluster_colors, ax = ax)
                ax.tick_params(labelsize=14)
                ax.set_ylabel('', fontdict= {'fontsize': 16, 'fontweight':'bold'})
                ax.set_xlabel('Percentage (%)',fontdict= { 'fontsize': 16, 'fontweight':'bold'})
                fig.savefig(buf, format="png", bbox_inches = "tight")
                st.image(buf)

        # Display row-normalized interaction matrix
        st.markdown('<p class="big-font">Interaction matrix (row-wise normalized)</p>', unsafe_allow_html=True)
        data_container = st.container()
        with data_container:
            table, plot, _ , _ = st.columns(4)
            with table:
                st.table(data=style_table(im_mtx_row))
            with plot:
                buf = BytesIO()
                fig, ax = plt.subplots()
                sns.heatmap(im_mtx_row, ax = ax)
                #ax.tick_params(labelsize=12)
                fig.savefig(buf, format="png", bbox_inches = "tight")
                st.image(buf)
        
        # Display slide-normalized interaction matrix
        st.markdown('<p class="big-font">Interaction matrix (slide-wise normalized)</p>', unsafe_allow_html=True)
        data_container = st.container()
        with data_container:
            table, plot, _ , _ = st.columns(4)
            with table:
                st.table(data=style_table(im_mtx_slide))
            with plot:
                buf = BytesIO()
                fig, ax = plt.subplots()
                sns.heatmap(im_mtx_slide, ax = ax)
                #ax.tick_params(labelsize=12)
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
                sns.barplot(data = df_cluster, y = 'Subtype', x = 'cluster_coeff' , palette = cluster_colors, ax = ax)
                ax.tick_params(labelsize=14)
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
        
        with st.spinner('Predicting aggressive scores...'):
            results = predict_survival(model, dataloader, device=device)
        config['label_column'] = 'risk_score'
      

        with st.spinner('Generating visualization...'):
            heatmap = generate_heatpmap_survival(slide, patch_size= (112,112), 
                                                results=results, 
                                                config = config)

        legend = Image.open('pictures/risk_score_legend.png')
        st.image(legend)
        im = Image.fromarray(heatmap)
        st.image(im, caption='Aggressive score prediction')

    if clear_button:
        clear(path)



        # # Display statistic tables for degree centrality
        # st.markdown('<p class="big-font">Degree centrality</p>', unsafe_allow_html=True)
        # data_container = st.container()
        # with data_container:
        #     table, plot, _ , _ = st.columns(4)
        #     with table:
        #         st.table(data=style_table(dgr_centr))
        #     with plot:
        #         buf = BytesIO()
        #         fig, ax = plt.subplots()
        #         sns.barplot(data = dgr_centr, y = 'Subtype', x = 'centrality' , palette = cluster_colors, ax = ax)
        #         ax.tick_params(labelsize=14)
        #         ax.set_ylabel('', fontdict= {'fontsize': 16, 'fontweight':'bold'})
        #         ax.set_xlabel('Centrality score',fontdict= { 'fontsize': 16, 'fontweight':'bold'})
        #         fig.savefig(buf, format="png", bbox_inches = "tight")
        #         st.image(buf)