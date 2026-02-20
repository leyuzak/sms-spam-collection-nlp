import json
import re
import string
from pathlib import Path

import joblib
import streamlit as st


# -------------------------
# text cleaning (training ile birebir)
# -------------------------
def clean_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"http\S+|www\.\S+", " ", x)
    x = re.sub(r"\S+@\S+", " ", x)
    x = re.sub(r"\d+", " ", x)
    x = x.translate(str.maketrans("", "", string.punctuation))
    x = re.sub(r"\s+", " ", x).strip()
    return x


# -------------------------
# load model artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent

    model = joblib.load(base / "model.joblib")
    vectorizer = joblib.load(base / "vectorizer.joblib")

    with open(base / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {int(v): k for k, v in label2id.items()}
    return model, vectorizer, label2id, id2label


# -------------------------
# prediction
# -------------------------
def predict_message(text, model, vectorizer, id2label):
    cleaned = clean_text(text)
    x_vec = vectorizer.transform([cleaned])

    pred_id = int(model.predict(x_vec)[0])
    pred_label = id2label[pred_id]

    spam_proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_vec)[0]
        spam_proba = float(probs[1])

    return pred_label, spam_proba, cleaned



st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Detector")
st.write(
    "This app classifies SMS messages as **Spam** or **Ham** using "
    "**TF-IDF + Logistic Regression**."
)

model, vectorizer, label2id, id2label = load_artifacts()

default_text = "Congratulations! You won a free prize. Call now!"
text = st.text_area("Enter SMS message", value=default_text, height=140)

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("Predict", type="primary")
with col2:
    show_clean = st.checkbox("Show cleaned text")

if predict_btn:
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        label, spam_proba, cleaned = predict_message(
            text, model, vectorizer, id2label
        )

        if label == "spam":
            st.error("🚨 Prediction: **SPAM**")
        else:
            st.success("✅ Prediction: **HAM**")

        if spam_proba is not None:
            st.metric("Spam probability", f"{spam_proba:.3f}")

        if show_clean:
            st.code(cleaned)

st.divider()


st.subheader("Batch prediction")
st.write("Enter multiple messages (one per line):")

batch_text = st.text_area(
    "Batch input",
    height=160,
    placeholder="Message 1\nMessage 2\nMessage 3"
)

if st.button("Predict batch"):
    lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
    if not lines:
        st.warning("No messages found.")
    else:
        cleaned = [clean_text(l) for l in lines]
        x_vec = vectorizer.transform(cleaned)
        preds = model.predict(x_vec)

        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_vec)[:, 1]

        rows = []
        for i, txt in enumerate(lines):
            row = {
                "text": txt,
                "prediction": id2label[int(preds[i])]
            }
            if probs is not None:
                row["spam_probability"] = float(probs[i])
            rows.append(row)

        st.dataframe(rows, use_container_width=True)
