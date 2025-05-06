# Sentiment Analysis App using Streamlit

This project provides a sentiment analysis application powered by machine learning models, including fine-tuned **DistilBERT**, **Logistic Regression**, **Multinomial Naive Bayes**, and **SVC**. Users can upload text or datasets to analyze sentiment interactively.

---

## **Project Overview**

The repository includes:
1. **Streamlit App** (`app.py`): Interactive web application for sentiment analysis.
2. **Model Training Notebooks**:
   - `sentiment_model.ipynb`: Covers basic model training using Logistic Regression, Naive Bayes, and SVC.
   - `Fine-TunedDistilBERT.ipynb`: Fine-tunes DistilBERT for sentiment analysis.
3. **Pretrained Models**:
   - Fine-tuned DistilBERT model stored in `Fine-TunedModel/`.

---

## **How to Set Up**

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```
### 2. **Set Up Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```
### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
### 4. **Run the Streamlit App**
```bash
streamlit run app.py
```

---

## **Acknowledgements**

This project leverages the following resources and tools:

- [Hugging Face Transformers](https://huggingface.co/): For fine-tuning the DistilBERT model.
- [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data): A dataset of labeled text data for sentiment analysis, sourced from Kaggle.
- [Streamlit](https://streamlit.io/): For building the interactive web application.
- [scikit-learn](https://scikit-learn.org/): For implementing classic machine learning models and performance evaluation.
- [Theory](https://www.analyticsvidhya.com/blog/2022/01/knowledge-distillation-theory-and-end-to-end-case-study/): For theory

Special thanks to the authors and contributors of these open-source tools and datasets for making this project possible.
