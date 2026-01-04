# 🤖 Intelligent Customer Support AI

**End-to-End NLP System using BERT + GPT (LoRA)**

---

## 📌 Project Overview

This project implements an **end-to-end intelligent customer support system** that automatically:

1. **Classifies customer feedback** using a fine-tuned **BERT encoder**
2. **Generates professional support responses** using a **GPT-style decoder fine-tuned with LoRA**
3. Serves predictions through an **interactive Streamlit web application**

The system demonstrates **real-world Generative AI engineering**, combining **encoder–decoder architectures**, **parameter-efficient fine-tuning**, and **production-oriented design**.

---

## 🧠 System Architecture

```
Customer Review
      ↓
BERT Sentiment Classifier
      ↓
Predicted Category
      ↓
GPT (LoRA Fine-Tuned)
      ↓
Automated Support Response
```

---

## 🔧 Models Used

### 🔹 BERT – Text Classification

* Model: `bert-base-uncased`
* Task: Sentiment Classification (Negative / Neutral / Positive)
* Training:

  * Hugging Face `Trainer`
  * Cross-entropy loss
  * Weighted F1-score & Accuracy
* Dataset:

  * Amazon Fine Food Reviews (processed)

### 🔹 GPT – Response Generation

* Base Model: `gpt2`
* Fine-Tuning Technique: **LoRA (PEFT)**
* Trainable Parameters: **~0.23%**
* Prompt-based generation:

  ```
  Customer Issue: {review}
  Category: {label}
  Support Response:
  ```

---

## ⚙️ Key Features

* ✅ Encoder vs Decoder Transformers (BERT vs GPT)
* ✅ LoRA fine-tuning (memory-efficient)
* ✅ Hugging Face Transformers & PEFT
* ✅ End-to-End NLP Pipeline
* ✅ Streamlit Web App
* ✅ CPU / Apple MPS compatible

---

## 📂 Project Structure

```
intelligent-customer-support-ai/
│
├── src/
│   ├── preprocess.py        # Data cleaning & labeling
│   ├── train_bert.py        # BERT sentiment training
│   └── train_gpt_lora.py    # GPT LoRA fine-tuning
│
├── app/
│   ├── pipeline.py          # End-to-end inference pipeline
│   └── streamlit_app.py     # Web application
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 🖥️ Demo

The application allows users to:

* Enter customer feedback
* View predicted sentiment
* Receive an AI-generated support response in real time

---

## 📈 Skills Demonstrated

* Transformer architectures (BERT & GPT)
* Generative AI engineering
* Parameter-efficient fine-tuning (LoRA)
* NLP pipelines & inference optimization
* Model deployment with Streamlit

---

## 🧑‍💻 Author

**Emad Qudah**
AI / Machine Learning Engineer
GitHub: [https://github.com/3madQudah](https://github.com/3madQudah)
