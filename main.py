import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# load imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

# load the pre-trained model with Relu activation
# model = load_model('simple_RNN_imdb.h5')
model = load_model('model.keras')
# model.save('model.keras')


# step2 helper function 
# function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# design the streamlit app

import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive and negative. ")

user_input = st.text_area("Movie Review")


if st.button('classify'):
    preprocess_input = preprocess_text(user_input)
    
    # make the prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    
    # display the result
    st.write(f"sentiment : {sentiment}")
    st.write(f"predict score : {1-prediction[0][0]}")
else:
    st.write("Please enter a moview review!")  

