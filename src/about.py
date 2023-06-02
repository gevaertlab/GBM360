"""
The main page of GBM360
"""

import streamlit as st
from ga import trace
import base64


def app():
    with st.container():
        st.markdown('<p class="big-font"> GBM360 is a machine learning tool that analyzes histology images to unravel the </p>', unsafe_allow_html=True)
        st.markdown('<p class="big-font"> spatial heterogeneity and predict the aggressiveness of glioblastoma cells.</p>', unsafe_allow_html=True)
        st.image('pictures/demo.png', width=1000)

    with st.expander("Disclaimer"):
        st.markdown("""GBM360 is an academic research project and should **not** be considered a medical device approved by any federal authorities.""")
    
    with st.expander("Contact"):
        st.markdown("""Visit us at: [Dr.Gevaert lab](https://med.stanford.edu/gevaertlab.html)""")
    
    trace()
    st.write("(c) 2023 All rights reserved.")
