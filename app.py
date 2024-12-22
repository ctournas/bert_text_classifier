import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('tournas/FineTuneBert')
    return tokenizer, model

tokenizer, model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button('Analyze')

d = {

    1: 'Toxic',
    0: 'Non Toxic'
}

if user_input and button:
    st.write('Analyzing...please wait.')

    # Tokenization with padding and truncation
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Get output from the model
    output = model(**test_sample)

    # Softmax for probabilities
    probs = torch.softmax(output.logits, dim=-1).detach().numpy()

    # Prediction: Find the class with the highest probability
    y_pred = np.argmax(probs, axis=1)
    st.write('prediction: ',d[y_pred[0]])
