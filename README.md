# Fine-Tuned Flan-T5 Summarization

## 📌 Project Overview
This project fine-tunes **Flan-T5**, a powerful text-to-text model by Google, on the **CNN/DailyMail dataset** to perform **text summarization**. The fine-tuned model generates concise summaries from long-form news articles with improved coherence and readability.

## 🚀 Datasets Used
### 1️⃣ **Tokenized CNN/DailyMail Dataset**
- **Description**: The dataset contains preprocessed and tokenized versions of the CNN/DailyMail dataset, used for training.
- **Link**: [Tokenized Dataset on Kaggle](https://www.kaggle.com/datasets/ayaanakhter/tokenized-dataset) (I crated the tokenized data from the cnn dataset and hosted it on kaggle for future use) 

### 2️⃣ **Fine-Tuned Flan-T5 Model**
- **Description**: The Flan-T5 model fine-tuned on the CNN/DailyMail dataset for text summarization.
- **Link**: [Fine-Tuned Flan-T5 on Kaggle](https://www.kaggle.com/datasets/ayaanakhter/fine-tuned-flan-t5) 

## 🛠️ Model Training Pipeline
1. **Dataset Preprocessing**
   - The CNN/DailyMail dataset was tokenized using the `T5Tokenizer`.
   - The dataset was split into **train, validation, and test** sets.

2. **Fine-Tuning Flan-T5**
   - The `T5ForConditionalGeneration` model was trained using **Hugging Face Transformers**.
   - Optimized using **AdamW** optimizer.
   - Training performed on **Kaggle GPU** with multiple epochs.

3. **Saving & Uploading the Model**
   - After training, the model was saved and uploaded to Kaggle as a dataset.

## 🌍 Hosting & Deployment
The model is deployed using **Gradio**, a simple web-based UI for testing text summarization.

### 🔹 How to Run the Gradio App on Kaggle
Run the following code inside a **Kaggle Notebook**:

```python
!pip install gradio

import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ✅ Load Model from Kaggle Dataset
model_path = "/kaggle/input/fine-tuned-flan-t5"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# ✅ Define Summarization Function
def summarize_text(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ Launch Gradio UI
interface = gr.Interface(fn=summarize_text, inputs="text", outputs="text", title="Flan-T5 Summarization")
interface.launch(share=True)  # Generates a public link
```

## 🎯 Key Features
✔ **Fine-tuned Flan-T5** for text summarization
✔ **Preprocessed CNN/DailyMail dataset** for training
✔ **Gradio UI for easy interaction**
✔ **Hosted on Kaggle (No Hugging Face required)**

## 🔗 Useful Links
- **Tokenized Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/ayaanakhter/tokenized-dataset)
- **Fine-Tuned Model**: [Kaggle Link](https://www.kaggle.com/datasets/ayaanakhter/fine-tuned-flan-t5)
- **Gradio Documentation**: [https://gradio.app](https://gradio.app)

## 📝 Author
**Ayaan Akhter** – [Kaggle Profile](https://www.kaggle.com/ayaanakhter)

