
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "bert_sentiment_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

st.title("BERT Sentiment Analyzer")
st.write("Classify a movie review as positive or negative.")

review = st.text_area("Enter a movie review")

if st.button("Analyze sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        tokenizer, model = load_model()
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        prediction = int(torch.argmax(probabilities))
        label = "Positive" if prediction == 1 else "Negative"
        st.subheader(f"Sentiment: {label}")
        st.write(f"Confidence: {probabilities[prediction].item():.1%}")
