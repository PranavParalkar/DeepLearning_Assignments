from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = FastAPI()

model = load_model("lstm_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
# Use model input shape so API padding always matches training.
max_seq_len = int(model.input_shape[1]) + 1

@app.get("/")
def home():
    return {"message": "LSTM Text Predictor API"}

@app.get("/predict/")
def predict(text: str):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

    predicted = model.predict(token_list)
    predicted_index = int(np.argmax(predicted, axis=-1)[0])  # FIX

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    return {"input": text, "next_word": output_word}