import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import streamlit as st
import numpy as np

# Loading the model
model = load_model("next_word_lstm.h5")

# Loading the tokenizer

with open("tokenizer.pkl","rb") as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_seq_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list)>=max_seq_len:
        token_list = token_list[-(max_seq_len-1):]

    token_list = pad_sequences([token_list],padding="pre",maxlen=max_seq_len)
    predicted = model.predict(token_list,verbose=0)
    predict_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predict_word_index:
            return word
    return None

st.title("Predict the next word using LSTM")

input_text = st.text_input("Enter the text","To be or not to be")

if st.button("Predict next word"):
    max_seq_len = model.input_shape[1]+1

    next_predicted_word = predict_next_word(model,tokenizer,input_text,max_seq_len)

    st.write(f"Next Word: {next_predicted_word}")