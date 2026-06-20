import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import torch.nn.functional as F
import nltk
import re

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

THRESHOLD = 0.62
MODEL_NAME = "Sanneeeeeeeee/robert-blurb-classifier"
TOKENIZER_NAME = "pdelobelle/robbert-v2-dutch-base"

@st.cache_resource
def load_model():
    login(token=st.secrets["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def classify(sentences, tokenizer, model):
    results = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        prob = F.softmax(outputs.logits, dim=1)[0][1].item()
        label = "Non-content" if prob >= THRESHOLD else "Content"
        results.append({"Sentence": sentence, "P(content)": round(prob, 3), "Label": label})
    return results

st.title("📚 Dutch Blurb Classifier")
st.write("Paste a Dutch book blurb below. Each sentence will be classified as **content** or **non-content**.")

blurb = st.text_area("Dutch blurb", height=200)

if st.button("Classify"):
    if not blurb.strip():
        st.warning("Please enter a blurb.")
    else:
        tokenizer, model = load_model()
        sentences = split_sentences(blurb)
        results = classify(sentences, tokenizer, model)
        
        for r in results:
            color = "#378ADD" if r["Label"] == "Content" else "#D85A30"
            st.markdown(
                f"<div style='padding:8px; margin:4px 0; border-left: 4px solid {color}; background:#000000'>"
                f"<b style='color:{color}'>{r['Label']}</b> (p={r['P(content)']})<br>{r['Sentence']}"
                f"</div>",
                unsafe_allow_html=True
            )
