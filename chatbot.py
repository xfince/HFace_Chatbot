import os
from dotenv import load_dotenv
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API key from the environment variable
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Define model names mapping if needed
MODEL_NAMES = {
    "llama-2-7b": "your-huggingface-model-name-for-llama-2-7b",
    "llama-2-13b": "your-huggingface-model-name-for-llama-2-13b",
    "llama-2-30b": "your-huggingface-model-name-for-llama-2-30b"
}

# Function to call the Hugging Face API
def call_llama_model(prompt, model_name):
    try:
        model_name_hf = MODEL_NAMES[model_name]
        api_key = HUGGINGFACE_API_KEY

        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, use_auth_token=api_key)
        model = AutoModelForCausalLM.from_pretrained(model_name_hf, use_auth_token=api_key)

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, use_auth_token=api_key)
        response = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]

        return response
    except Exception as e:
        return f"Error: {e}"

# Streamlit app interface
st.title("Buddy your Blog Bot")
st.write("Buddy runs on the Hugging Face model.")

# Hugging Face model selection drop-down menu
model_name = st.selectbox("Select LLaMA model", ["llama-2-7b", "llama-2-13b", "llama-2-30b"])

# Initialize chat history if not already in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input field
user_input = st.text_input("You:", "", placeholder="Tell me Something About Travel")

# Send button and processing the user input
if st.button("Send"):
    if user_input:
        # Create prompt by combining pre-prompt with user input
        PRE_PROMPT = "You are a helpful personal assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as a Personal Assistant."
        prompt = f"{PRE_PROMPT} {user_input}"
        # Call the Hugging Face model and get the response
        response = call_llama_model(prompt, model_name)
        # Update the chat history with user input and model response
        st.session_state['chat_history'].append(("User", user_input))
        st.session_state['chat_history'].append(("Assistant", response))

# Display chat history
for speaker, text in st.session_state['chat_history']:
    st.write(f"**{speaker}**: {text}")

# Function to open the link back to the blog
def open_blog():
    webbrowser.open_new_tab("https://www.google.com/index.html")

st.sidebar.button("Back to Blog", on_click=open_blog)
