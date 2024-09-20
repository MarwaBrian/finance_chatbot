import pandas as pd
import spacy
import transformers 
import requests
import json
from bs4 import BeautifulSoup
import streamlit as st 
import os
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM

# API_URL = "https://api-inference.huggingface.co/models/huggyllama/llama-7b" model was too big to be loaded via the standard api
token_chacha = os.getenv('HF_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"
headers = {"Authorization": f"Bearer {token_chacha}"}

# Function to query Hugging Face API
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Function to process and return response
def get_response(input_text):
    response = query({"inputs": input_text})
    # Extract generated text from response JSON
    return response [0]['generated_text'] if 'generated_text' in response[0] else response

#


# get_response('which kind of credit cards does kenya commmercial bank offer?')


# Streamlit app layout
st.title("Kenya Finance Chatbot")
st.write("Ask anything about banks and financial services in Kenya!")

# Input box for the user's question
user_input = st.text_input("Enter your question:")

# When the user clicks the 'Submit' button
if st.button("Submit"):
    if user_input:
        with st.spinner('Thinking...'):
            # Get the response from the chatbot
            response = get_response(user_input)
        # Display the response
        st.write("Chatbot: ", response)
    else:
        st.write("Please enter a question.")

# About section
st.sidebar.title("About")
st.sidebar.info("""
This finance chatbot provides information on banks in Kenya. Powered by Hugging Face's LLaMA model.
""")