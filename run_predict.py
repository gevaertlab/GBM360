import numpy as np
from PIL import Image
import streamlit as st
from openslide import OpenSlide
from heatmap_survival import generate_heatpmap_survival
import pandas as pd

from utils import *
from spa_mapping import generate_heatmap
from heatmap_survival import generate_heatpmap_survival
import seaborn as sns
import matplotlib.pyplot as plt

def app():

    # Main config
    config = {
    "model_name": "resnet50",
    "target_label": "max_class",
    "use_h5": 0, 
    "num_classes": 6,
    "batch_size": 128,
    "use_cuda":True,
    'label_column' : 'label',
    'pretrained': True,
    "img_size": 46,
    'aggregator': 'identity',
    "aggregator_hdim": 2048,
    "max_patch_per_wsi_train" :100000,
    "max_patch_per_wsi_val" : 100000,
    "compress_factor": 16
    }

    # Specify canvas parameters in application
    bg_image = st.file_uploader("Image:", type=["tiff","svs"])

    # TODO max width / height
    #stroke_color = st.sidebar.color_picker("Box border color: ")
    stroke_color = "#000"
    test_mode = st.selectbox("Test mode (only 1000 patches will be predicted):", ("True", "False"))
    cell_type_button = st.button("Get cell type visualization")
    prognosis_button = st.button("Get prognosis visualization")
    clear_button = st.button("Clear the session")

    device = check_device(config['use_cuda'])
    
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
        st.markdown('<p class="big-font">Upload an image first</p>', unsafe_allow_html=True)

    # https://stackoverflow.com/questions/66372402/conversion-of-dimension-of-streamlit-canvas-image
    # https://github.com/andfanilo/streamlit-drawable-canvas/issues/39

    max_patches_per_slide = np.inf
    if test_mode == "True":
        max_patches_per_slide = 1000

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
        legend = Image.open('pictures/cell-type-hor.png')
        st.image(legend)
        st.image(im, caption='Cell distribution accross the tissue')


        # Display statistic tables for cell proportions 
        st.markdown('<p class="big-font">Percentage of each cell type in the slide</p>', unsafe_allow_html=True)
        df = pd.DataFrame([percentages])
        df = df.T
        df.columns = ['Percentage (%)']
        df['Cell type'] = df.index
        df = df[['Cell type', "Percentage (%)"]]
        df = df.reset_index(drop = True)

        # fig = plt.figure(figsize=(1,1))
        # sns.barplot(data = df, x = "Cell type", y = "Percentage (%)")
        # st.pyplot(fig)

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
        <body>
        <html>
        '''
        table_html = df.style.set_properties(
        subset=['Cell type'], 
        **{'font-weight': 'bold', 'text-align': 'center'}).render()
        #st.table(df)
        st.write(html_string.format(table=table_html), unsafe_allow_html=True)
        #st.image([im, legend], caption=['Cell distribution accross the tissue', None])

    if prognosis_button:
        with st.spinner('Reading patches...'):
            dataloader = read_patches(slide, max_patches_per_slide)

        config['num_classes'] = 1
        with st.spinner('Loading model...'):
            model = load_model(checkpoint='model_survival.pt', config = config)
        
        with st.spinner('Predicting survival...'):
            results = predict_survival(model, dataloader, device=device)

        config['label_column'] = 'risk_score'
        with st.spinner('Generating visualizations...'):
            heatmap = generate_heatpmap_survival(slide, patch_size= (112,112), 
                                                results=results, 
                                                config = config)

        im = Image.fromarray(heatmap)
        st.image(im, caption='Survival prediction accross the tissue')

    if clear_button:
        clear(path)