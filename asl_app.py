import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

MODEL_PATH = 'my_model.keras'
model = load_model(MODEL_PATH)
IMAGE_SIZE = (64, 64)


## LABEL MAP ##
with open('label_map.json') as f:
    label_map = json.load(f)
## END LABEL MAP ##


def predict_class(img, model):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return label_map[predicted_class[0]]

st.title('ASL Alphabet Recognition')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=IMAGE_SIZE)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying in progres...")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  #normalise img array
    predicted_batch = model.predict(img_array)
    predicted_class = np.argmax(predicted_batch, axis=1)
    predicted_label = label_map[str(predicted_class[0])]  #ensure label_map uses string keys
    st.write(f"The predicted sign is: {predicted_label}")




