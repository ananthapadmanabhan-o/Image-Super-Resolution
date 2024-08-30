import streamlit as st 

from srgan.utils import predict 

st.set_page_config(
    page_title='Demo',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.header('Image Super Resolution GAN',divider='rainbow',)

uploaded_img = st.file_uploader('',type=['png','jpeg','bmp','jpg'])

if uploaded_img is not None:

    l_box, r_box = st.columns(2)

    with l_box:
        st.image(image=uploaded_img,use_column_width=True)

    if st.button('Generate',type='primary'):
        out_image = predict(uploaded_img,'models/srgan6_4x.pth','cuda')
        with r_box:
            st.image(out_image,use_column_width=True)
