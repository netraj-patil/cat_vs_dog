# import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
import streamlit as st

# print(cv2.__version__)

st.title("Cat vs Dog Image Classification")

model = load_model('model.h5')

# Load image
img = st.file_uploader("Choose a file : ", type='jpeg')

if img is not None:
    img = image.load_img(img,target_size=(256,256))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    st.image(img)

# Make prediction
generate_pred=st.button("Generate Prediction")
if generate_pred:
    predictions = model.predict(img_array)
    if predictions[0]:
        st.title("Dog")
    else:
        st.title("Cat")
