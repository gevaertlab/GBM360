"""
The main page of GBM360
"""

import streamlit as st
from ga import trace
import base64


def app():
    with st.container():
        st.markdown('<p class="big-font"> GBM360 is a software that harnesses the power of machine learning to investigate the cellular heterogeneity and spatial architecture of glioblastoma </p>', unsafe_allow_html=True)
        st.image('pictures/demo.png', width=1000)

    with st.expander("Disclaimer"):
        st.markdown("""GBM360 is an academic research project and should **not** be considered a medical device approved by any federal authorities.""")
    
    with st.expander("Contact"):
        st.markdown("""Visit us at: [Dr.Gevaert lab](https://med.stanford.edu/gevaertlab.html)""")
    
    trace()
    st.write("(c) 2023 All rights reserved.")
