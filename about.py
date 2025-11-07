import streamlit as st
from data.base import about_pharmacy_management, warn_pharmacy_system

def app():
    st.markdown(about_pharmacy_management)
    st.warning(warn_pharmacy_system)
