import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
import streamlit as st


# title
st.title("Cat vs Dog Image Classification")

model = load_model('model2.h5')

# Load image
img = st.file_uploader("Choose a file : ", type='jpeg')

if img is not None:
    img = image.load_img(img,target_size=(256,256))
    img_array=image.img_to_array(img)
    img_array = tf.cast(img_array/255, tf.float32)
    img_array=np.expand_dims(img_array,axis=0)
    st.image(img)

# Make prediction
generate_pred=st.button("Generate Prediction")
if generate_pred:
    
    predictions = model.predict(img_array)
    # check probability of being a dor and cat
    # if probability>0.5 then it is dog else it is cat
    # here we have chosen threshold as 50%
    if predictions > 0.5:
        st.title("It is a Dog")
    else:
        st.title("It is a Cat")
