import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import torch.nn.functional as F
import re

st.markdown("""
<style>
.block-container {
    padding-top: 1.7rem;
}
p {
    line-height: 1.2;
    padding-left: 0.2rem;
}
.stButton > button {
    background-color: #ffffff !important;
}
.stTextArea textarea::placeholder {
    color: #999999 !important;
    font-style: italic !important;
}
div[data-testid="stTextArea"] {
    margin-top: -30px;
}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 0.62
MODEL_NAME = "Sanneeeeeeeee/robert-blurb-classifier"
TOKENIZER_NAME = "pdelobelle/robbert-v2-dutch-base"

EXAMPLES = [
    {
         "nl": "Wanneer Nova op een onbekende plaats ontwaakt, weet ze niet meer wie ze is, noch waar ze vandaan komt.",
        "en": "When Nova wakes up in an unknown place, she no longer knows who she is or where she came from.",
        "expected": "Content"
    },
    {
        "nl": "Sinds haar jeugd is ze verslingerd aan boeken.",
        "en": "Since her childhood, she has been passionate about books.",
        "expected": "Non-content"
    },
    {
        "nl": "De pers over Julia Navarro: 'Een grote roman' de Volkskrant 'Over romantiek in tijden van ellende.'",
        "en": "The press about Julia Navarro: 'A great novel' de Volkskrant 'About romance in times of misery.'",
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

st.markdown("<h3 style='margin-bottom: 0rem;font-weight:bold;'>Dutch Blurb Sentence Classifier</h3>", unsafe_allow_html=True)

st.markdown("<p style='font-weight:bold;margin-bottom:+4px;font-size:1.2em;'>Try a real blurb sentence:</p>", unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with cols[i]:
        if st.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state["input_sentence"] = ex["nl"]
        st.markdown("<p style='font-size:1em;font-style: italic;margin-top:-10px;'>"+ex["en"]+"</p>", unsafe_allow_html=True)

st.markdown("<p style='font-weight:bold;margin-bottom:+2px;font-size:1.2em;'>Or type your own:</p>", unsafe_allow_html=True)

sentence = st.text_area(
    "",
    value=st.session_state.get("input_sentence", ""),
    height=100,
    key="input_sentence",
    placeholder="Type a Dutch sentence that could appear in a blurb (content or non-content) or select an example"
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
