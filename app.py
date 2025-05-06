import streamlit as st
import pandas as pd
import joblib
import chardet
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the fine-tuned model
DistilBERT = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
DistilBERT.load_state_dict(torch.load('./Fine-TunedModel/model_state.pth', map_location=torch.device('cpu')))

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./Fine-TunedModel')

# Load vectorizer and model
loaded_objects = joblib.load('sentiment_model.pkl')
vectorizer = loaded_objects['vectorizer']
LogisticRegression = loaded_objects['LogisticRegression']
MultinomialNB = loaded_objects['MultinomialNB']
SVM = loaded_objects['SVM']
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

models = {
    "SVM": SVM,
    "Multinomial NB": MultinomialNB,
    "Logistic Regression": LogisticRegression,
    "DistilBERT": DistilBERT
}

# Function for sentiment analysis
def analyze_sentiment(text, model):
    # Vectorize the input text
    transformed_text = vectorizer.transform([text])
    # Predict sentiment
    return model.predict(transformed_text)[0] 

# Function for DistilBERT sentiment analysis
def analyze_sentiment_fineTuned(text ,model):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return label_map[predictions.item()]

# Function for model evaluation

def get_predictions(_model, model_name, _X_test):
    if model_name == 'DistilBERT':
        y_pred = _X_test.apply(lambda x: analyze_sentiment_fineTuned(x, _model))
    else:
        y_pred = _X_test.apply(lambda x: analyze_sentiment(x, _model))
    return y_pred

def evaluate_model(model, X_test, y_test, model_name):
    st.subheader(f"Evaluation of {model_name}")
    
    y_pred = get_predictions(model, model_name, X_test)

    # Classification report
    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.text("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# Streamlit app
st.title("Sentiment Analysis App")

st.markdown(f"""
This model was trained on a dataset available at: 
[Dataset Link]({"https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data"})""")
# Single sentence analysis
st.header("Single Sentence Analysis")
model_name = st.selectbox("Select Model", ("Logistic Regression","Multinomial NB", "SVM", "Fine-Tuned_DistilBERT"))
sentence = st.text_input("Enter a sentence:")
if st.button("Analyze Sentence"):
    if sentence:
        if model_name =='Multinomial NB':
            result = analyze_sentiment(sentence, models['Multinomial NB'])
        if model_name == 'Logistic Regression':
            result = analyze_sentiment(sentence, models['Logistic Regression'])
        if model_name == 'SVM':
            result = analyze_sentiment(sentence, models['SVM'])
        if model_name == 'Fine-Tuned_DistilBERT':
            result = analyze_sentiment_fineTuned(sentence, models['DistilBERT'])
        st.write(f"Sentiment: {result}")
    else:
        st.write("Please enter a sentence.")

# CSV file analysis
st.header("CSV File Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    data.dropna(subset= ['text'], inplace=True)

    if 'text' in data.columns:
        # Vectorize and analyze sentiment for each row
        if model_name == 'Fine-Tuned_DistilBERT':
            data['sentiment'] = data['text'].apply(lambda x: analyze_sentiment_fineTuned(x, models['DistilBERT']))
        else:
            data['sentiment'] = data['text'].apply(lambda x: analyze_sentiment(x, models[model_name]))

        st.write(data)

        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Analyzed CSV",
            data=csv,
            file_name="analyzed_sentiments.csv",
            mime="text/csv"
        )

        sentiment_counts = data['sentiment'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Sentiment Labels")

        st.pyplot(fig)
    else:
        st.write("The CSV file must contain a 'text' column.")
else:
    st.write("Please upload a CSV file with 'text' field.")

# Model Evaluation
st.header("Model Evaluation Section")

data = pd.read_csv("./Data/test.csv")
data.dropna(subset=['text', 'sentiment'], inplace=True)
X_test = data['text']
y_test = data['sentiment'] 

selected_model = st.selectbox("Select Model ", options=list(models.keys()))

if st.button("Evaluate Model"):
    if selected_model:
        model = models[selected_model]
        evaluate_model(model, X_test, y_test, selected_model)
    else:
        st.write("Please select a model before processing.")

# Footer function
def add_footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #000000;
            text-align: center;
            padding: 10px 0;
            font-size: 12px;
            color: #808080;
        }
        </style>
        <div class="footer">
            © 2024 Saksham Dura. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )
add_footer()