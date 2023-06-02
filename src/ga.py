"""
Google analytics
"""

import streamlit as st

def trace():
    st.markdown(
    """
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXE40WY0KS"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'G-XXE40WY0KS');
        </script>
    """, unsafe_allow_html=True)
    return 