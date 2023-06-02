"""
Landing point of GBM360
"""

import streamlit as st
import os
from utils import *
import run_predict
import auth
import about

st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
.streamlit-expanderHeader {
    font-size: x-large;
    font-weight: bold;
}
button[data-baseweb="tab"] {
  font-size: 26px;
  font-weight: bold;

}
</style>
""", unsafe_allow_html=True)

st.image("pictures/logo.png", width = 150)

tab1, tab2, tab3 = st.tabs(["About", "Tutorial", "Run"])

with tab1:
    about.app()

with tab2:
    #st.markdown('<p class="big-font">Registration is not required at this time.</p>', unsafe_allow_html=True)
    intro_markdown = read_markdown_file("tutorial.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

with tab3:
    # if not st.session_state["authentication_status"]:
    #     st.markdown('<p class="big-font">Please log in first.</p>', unsafe_allow_html=True)
    # else:
    if not os.path.exists('temp'):
        os.mkdir('temp')
    run_predict.app()
        
        
