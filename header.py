import streamlit as st
from data.base import head_pharmacy, st_style, footer_pharmacy


def app():
    st.markdown(st_style,
            unsafe_allow_html=True)

    st.markdown(footer_pharmacy,
                unsafe_allow_html=True)


    st.markdown(head_pharmacy,
        unsafe_allow_html=True
    )