# pip install -r requirements.txt
# streamlit run app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import pandas as pd
import numpy as np
from PIL import ImageOps
import keras
from keras.models import load_model
import os

model = tf.keras.models.load_model('model.h5')


def preprocess_image(image):
    """Preprocesses an image for model prediction."""
    image = Image.open(image)
    image = image.resize((224,224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    # image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    return image_array.reshape(1,224,224, 3)  # Add batch dimension

def predict_food(image_file):
    """Performs image preprocessing, prediction, and confidence score calculation."""
    preprocessed_image = preprocess_image(image_file)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class] * 100

    if predicted_class==0:
       food ='Baked potato'
    elif predicted_class==1:
       food ='Burger'
    elif predicted_class==2:
       food ='Crispy chicken'
    elif predicted_class==3:
        food ='Donut'
    elif predicted_class==4:
        food ='Fries'
    elif predicted_class==5:
        food ='Hot Dog'
    elif predicted_class==6:
        food ='Pizza'
    elif predicted_class==7:
        food ='Sandwich'
    elif predicted_class==8:
        food ='Taco'
    else:
        food ='Taquito'



    return food, confidence_score

st.set_page_config(
    page_title="Food Classification",
    page_icon="💡",
    # layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': '',
    #     'Report a bug': '',
    #     'About': ''
    # }
    )

gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">FOOD CLASSIFICATION</div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)
st.image(r'fd img.jpg', width=700)
st.subheader("**is this burger or something?** Upload an image to find out!")



uploaded_file = st.file_uploader("Choose an image:", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_food, confidence = predict_food(uploaded_file)
    st.success(f"Predicted food: {predicted_food}(: {confidence:.2f}%)")


    








