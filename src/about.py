"""
The main page of GBM360
"""

import streamlit as st
import base64


def app():

    with st.container():
        st.markdown('<p class="big-font"> GBM360 is a software that harnesses the power of machine learning to investigate the cellular heterogeneity and spatial architecture of glioblastoma </p>', unsafe_allow_html=True)
        st.image('pictures/demo.png', width=1000)

    with st.expander("Citation"):
        st.markdown("""Zheng, Y., Carrillo-Perez, F., Pizurica, M. et al. Spatial cellular architecture predicts prognosis in glioblastoma. Nat Commun 14, 4122 (2023). https://doi.org/10.1038/s41467-023-39933-0""")

    with st.expander("Disclaimer"):
        st.markdown("""GBM360 is an academic research project and should **not** be considered a medical device approved by any federal authorities.""")
        st.markdown("<span style='font-size: 18px;'>Please remove <a href='https://med.stanford.edu/irt/security/stanfordinfo/hipaa.html'>Personal Health Information (PHI)</a> from all uploaded files, as we are not responsible for data compliance issues</span>", unsafe_allow_html=True)
    
    with st.expander("Contact"):
        paragraph = "<span style='font-size: 18px;'>- Dr. Yuanning Zheng is a postdoctoral scholar at Stanford University. He obtained his PhD degree in Medical Sciences from Texas A&M University and a Master in Computer Science from Georgia Institute of Technology.</span>\n" \
                    "<span style='font-size: 18px;'>Dr. Zheng's research focuses on developing innovative machine learning and bioinformatics methods to unravel the heterogeneity and improve personalized diagnosis of cancers and other complex diseases. Email: eric2021@stanford.edu</span>\n\n" \
                    "<span style='font-size: 18px;'>- Dr. Olivier Gevaert is an associate professor at Stanford University focusing on developing machine-learning methods for biomedical decision support from multi-scale data. Email: ogevaert@stanford.edu</span>\n\n" \
                    "<span style='font-size: 18px;'>- Other contributors: Francisco Carrillo-Perez </span>\n\n" \
                    "<span style='font-size: 18px;'>- Visit us at: <a href='https://med.stanford.edu/gevaertlab.html'>Dr. Gevaert lab</a></span>\n\n" \
                    "<span style='font-size: 18px;'>- For bug reporting, please visit: <https://github.com/gevaertlab/GBM360></span>"

        st.markdown(paragraph, unsafe_allow_html=True)
    
    st.write("(c) 2023 All rights reserved.")

    