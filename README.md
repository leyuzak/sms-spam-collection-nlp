# 📩 SMS Spam Detection using NLP

🔗 **Live Demo (Hugging Face Spaces):**  
https://huggingface.co/spaces/leyuzak/SMS-Spam-Collection-using-NLP

🔗 **Kaggle Notebook:**  
https://www.kaggle.com/code/leyuzakoksoken/sms-spam-collection-using-nlp

---

## 📌 Project Overview

This project is an end-to-end **Natural Language Processing (NLP)** application designed to classify SMS messages as **Spam** or **Ham (Non-Spam)**.

The workflow covers the complete machine learning lifecycle, including:
- Exploratory Data Analysis (EDA)
- Text preprocessing
- Feature extraction
- Model training and interpretation
- Deployment as an interactive web application

---

## 🎯 Objective

The main objective is to automatically identify spam SMS messages based solely on their text content.  
Spam messages typically contain promotional, urgent, or deceptive language, while ham messages are conversational and personal.

---

## 📊 Dataset

- **Name:** SMS Spam Collection Dataset  
- **Classes:**
  - `ham` – legitimate messages
  - `spam` – unsolicited or fraudulent messages
- **Characteristics:**
  - Highly imbalanced class distribution
  - Spam messages are generally longer
  - Distinct vocabulary patterns between classes

---

## 🔎 Exploratory Data Analysis (EDA)

Key insights discovered during EDA:

- Strong class imbalance favoring ham messages
- Spam messages tend to have significantly greater text length
- Frequent spam keywords include *free*, *call*, *claim*, *win*
- Ham messages contain conversational words such as *ok*, *sorry*, *home*

These insights informed preprocessing and modeling decisions.

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied to the text data:

- Conversion to lowercase
- Removal of URLs, emails, numbers, and punctuation
- Whitespace normalization
- Generation of a cleaned text feature for modeling

---

## 🧠 Feature Engineering & Modeling

- **Text Representation:** TF-IDF Vectorization  
  - Unigrams and bigrams  
- **Classifier:** Logistic Regression  
- **Imbalance Handling:** Class weighting (`class_weight="balanced"`)

This approach provides:
- Strong baseline performance
- Fast training and inference
- High interpretability via model coefficients

---

## 📈 Model Interpretation

The trained model highlights meaningful linguistic patterns:

### Top Spam Indicators
- `call`, `free`, `claim`, `win`, `reply`, `mobile`

### Top Ham Indicators
- `ok`, `sorry`, `home`, `later`, `love`, `got`

These results align with real-world characteristics of spam and non-spam messages.

---

## 🖼️ Visualization

- Class distribution plots
- Text length distribution (ham vs spam)
- WordCloud visualizations for both classes
- Confusion matrix for model evaluation

---

## 🚀 Web Application

The trained model is deployed as a **Streamlit** application on **Hugging Face Spaces**.

### Features:
- 🔍 Single message classification
- 📊 Spam probability score
- 📄 Batch prediction (multiple messages)
- 🧹 Optional display of cleaned text

Access the live demo here:  
👉 https://huggingface.co/spaces/leyuzak/SMS-Spam-Collection-using-NLP

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Docker
- Hugging Face Spaces
- Kaggle

---

## 📌 Results

The model achieves strong performance, particularly in detecting spam messages when evaluated using precision, recall, and F1-score.

The results demonstrate that classical NLP techniques remain effective for text classification tasks.

---

## 🔮 Future Improvements

Potential enhancements include:
- Transformer-based models (e.g., BERT)
- Threshold tuning for improved spam recall
- REST API deployment (FastAPI)
- Multi-language SMS support

