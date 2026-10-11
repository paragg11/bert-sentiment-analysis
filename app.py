import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "paragg11/bert-sentiment-model"

st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

st.title("🎬 BERT Sentiment Analyzer")
st.write(
    "Enter a movie review to predict whether its sentiment "
    "is positive or negative."
)

with st.spinner("Loading BERT model..."):
    tokenizer, model = load_model()

review = st.text_area(
    "Your movie review",
    placeholder="Example: The story was excellent and the acting was brilliant!"
)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        prediction = int(torch.argmax(probabilities).item())
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        confidence = probabilities[prediction].item() * 100

        st.subheader(f"Predicted sentiment: {sentiment}")
        st.progress(float(probabilities[prediction].item()))
        st.write(f"Model confidence: {confidence:.2f}%")
        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )
