import streamlit as st

def app():
    with st.container():
        st.markdown('<p class="big-font">Intra-tumoral heterogeneity is a hallmark of glioblastoma (GBM). GBM360 is a machine learning tool to resolve the spatial distribution of GBM cells of different transcriptional subtypes. The model is trained using spatial transcriptomics data and predicts cellular states from histology images.</p>', unsafe_allow_html=True)
        st.image('pictures/demo.png', width=1000)
    
    with st.expander("Description of the transcriptional subtypes"):
        st.markdown("""* Normal: normal brain cells lack of chromosomal alterations.""")
        st.markdown("""* NPC-like: cancer cells highly expressed synaptic genes (*SYT1*, *SYN2*, *CALM3*) and markers for neural progenitor cells (*SNAP25*, *CD24* and *SYN1*).""")
        st.markdown("""* OPC-like: cancer cells highly expressed genes associated with oligodendrocyte progenitors, including *PLP1*, *CNP*, *MBP*.""")
        st.markdown("""* Reactive-immune: cancer cells highly expressed glial-related genes, including radial glial (*PTPRZ1*, *HOPX*) and astrocytic markers (*GFAP*, *APOC1*). These cells also express genes associated with immune response (*HLA-DRA*, *HLA-DRB*, *B2M*, *CD74*), likely reflects the reactive transformation of astrocytes.""")
        st.markdown("""* Reactive-hypoxia: cancer cells highly expressed mesenchymal-related genes (*VIM*) and genes associated with hypoxia response (*VEGFA*, *HILPDA*, *ADM*), glycolytic process (*GAPDH*, *PGK1*, *LDHA*), and response to reduced oxygen-levels (*NRDG1*, *MDM2*, *ERO1A*).""")
        st.markdown("""* Mesenchymal (MES): cancer cells highly expressed genes associated with extracellular matrix, including collagens (*COL6A1*, *COL3A1*), fibronectin (*FN1*), proteoglycan (*BGN*), and matrix metallopeptidase (*MMP2*, *MMP9*), but lacks hypoxia-related signatures.""")
     
    with st.expander("Clinical implications of the transcriptional subtypes"):
        st.markdown("""Application of GBM360 to histology images from the 389 patients of the TCGA-GBM cohort showed that the proportions of the NPC-like cells were associated with good patient survival, whereas the proportions of the reactive-hypoxia cells were both associated with worse patient survival.""")
    
    with st.expander("Disclaimer"):
        st.markdown("""GBM360 is developed during academic research. It is *not* a medical device approved by any federal authorities""")
    
    with st.expander("Contact"):
        st.markdown("""Visit us at: [Dr.Gevaert lab](https://med.stanford.edu/gevaertlab.html)""")
        
        
    st.write("(c) 2022 All rights reserved.")