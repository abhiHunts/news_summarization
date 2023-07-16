import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 model and tokenizer
model_name = 'abhi-pwr/news-summarizer'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set maximum sequence length for the input text
max_length = 512

# Function to generate the summary
def generate_summary(article):
    inputs = tokenizer.encode(article, return_tensors='pt', max_length=max_length, truncation=True)
    inputs = inputs.to(device)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Streamlit app
st.title("T5 Text Summarization")

# Text input box
article = st.text_area("Enter the news article", height=300)

# Generate summary button
if st.button("Generate Summary"):
    if article:
        summary = generate_summary(article)
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter a news article to generate a summary.")
