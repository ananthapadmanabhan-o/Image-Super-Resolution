import streamlit as st

st.set_page_config(
    page_title="Documentation",
    layout='centered',
    page_icon="📄",
    initial_sidebar_state="expanded",
)

st.header('Welcome to Documentation! 👋', divider='rainbow')

with open('README.md','r') as f:
    body = f.read()

st.markdown(body)