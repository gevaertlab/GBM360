import numpy as np
from PIL import Image
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_drawable_canvas import st_canvas
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

tab1, tab2, tab3 = st.tabs(["About", "Log in", "Run"])

with tab1:
    about.app()

with tab2:
    #auth.app()
    None

with tab3:
    # if not st.session_state["authentication_status"]:
    #     st.markdown('<p class="big-font">Please log in first.</p>', unsafe_allow_html=True)
    # else:
    if not os.path.exists('temp'):
        os.mkdir('temp')
    run_predict.app()
        
        
