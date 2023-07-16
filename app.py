import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the T5 model and tokenizer
model_name = "t5-base-medium-title-generation/checkpoint-1200"
model_dir = f"./checkpoint-1200-20230716T083021Z-001"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to('cpu')

# Set maximum sequence length for the input text
max_length = 512

# Function to generate the summary
def generate_summary(article):
    inputs = tokenizer.encode(article, return_tensors='pt', max_length=max_length, truncation=True)
    inputs = inputs.to(device)

    # Generate summary
    output = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    summary = '.\n'.join(decoded_output.split('. '))

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
