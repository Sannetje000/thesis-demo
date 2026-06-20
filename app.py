import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import torch.nn.functional as F
import re

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
p {
    line-height: 1.2;
    padding-left: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 0.62
MODEL_NAME = "Sanneeeeeeeee/robert-blurb-classifier"
TOKENIZER_NAME = "pdelobelle/robbert-v2-dutch-base"

EXAMPLES = [
    {
        "nl": "De twee broers groeien op in een klein dorp waar iedereen elkaar kent.",
        "en": "The two brothers grow up in a small village where everyone knows each other.",
        "expected": "Content"
    },
    {
        "nl": "Een meeslepend debuut van een veelbelovende nieuwe stem in de Nederlandse literatuur.",
        "en": "A gripping debut from a promising new voice in Dutch literature.",
        "expected": "Non-content"
    },
    {
        "nl": "Geïllustreerd door de bekroonde kunstenaar Jan de Vries.",
        "en": "Illustrated by award-winning artist Jan de Vries.",
        "expected": "Non-content"
    },
]

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
        results.append({"Sentence": sentence, "P(non-content)": round(prob, 3), "Label": label})
    return results

st.markdown("<h4.5 style='margin-bottom: 0rem;'>Dutch Blurb Sentence Classifier</h3>", unsafe_allow_html=True)

st.markdown("<p style='font-weight:bold;margin-bottom:+2px;font-size:1.1em;'>Try an example:</p>", unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with cols[i]:
        if st.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state["input_sentence"] = ex["nl"]
        st.markdown("<p style='font-size:1em; margin-top:-10px;'>"+ex["en"]+"</p>", unsafe_allow_html=True)

st.markdown("<p style='font-weight:bold; font-size:1.1em;'>Or enter your own Dutch sentence:</p>", unsafe_allow_html=True)

st.markdown("""
<p style='color:#000000; font-size:0.95em'>
<span style='color:#6AAAD4; font-size:1.2em'>●</span> <b style='color:#6AAAD4'>Content</b>: sentences describing the story, characters, or setting<br>
<span style='color:#E08070; font-size:1.2em'>●</span> <b style='color:#E08070'>Non-content</b>: promotional text, author credits, critic quotes, or other metadata
</p>
""", unsafe_allow_html=True)

sentence = st.text_area(
    "",
    value=st.session_state.get("input_sentence", ""),
    height=100,
    key="input_sentence"
)

if st.button("Classify"):
    if not sentence.strip():
        st.warning("Please enter a sentence.")
    else:
        tokenizer, model = load_model()
        sentences = split_sentences(sentence)
        results = classify(sentences, tokenizer, model)
        for r in results:
            color = "#378ADD" if r["Label"] == "Content" else "#D85A30"
            st.markdown(
                f"<div style='padding:12px; margin:6px 0; border-radius:8px; background:#ffffff; border: 1px solid #000000; color:#000000; text-align:center;'>"
                f"<b style='color:{color}; font-size:1.3em'>● {r['Label']} sentence</b><br/>"
                f"<span style='font-size:0.9em; color:#000000'>The model classified your sentence as <b>{r['Label'].lower()}</b>.</span><br/>"
                f"<span style='font-size:0.85em; color:#000000'>Non-content probability: {int(r['P(non-content)']*100)}% (threshold: 62%)</span>"
                f"</div>",
                unsafe_allow_html=True
            )
