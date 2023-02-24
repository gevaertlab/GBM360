"""
The main page of GBM360
"""

import streamlit as st
import base64


def app():
    with st.container():
        st.markdown('<p class="big-font"> GBM360 is a machine learning tool that uses histology images to resolve the spatial heterogeneity of glioblastoma cells.</p>', unsafe_allow_html=True)
        st.image('pictures/demo.png', width=1000)

    with st.expander("Disclaimer"):
        st.markdown("""GBM360 is developed from academic investigation. It is *not* a medical device approved by any federal authorities""")
    
    with st.expander("Contact"):
        st.markdown("""Visit us at: [Dr.Gevaert lab](https://med.stanford.edu/gevaertlab.html)""")
        
    st.write("(c) 2022 All rights reserved.")
