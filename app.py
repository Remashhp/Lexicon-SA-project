
# Project Title: Lexicon-Based Sentiment Analysis (SA) with VADER

# --- Library Imports ---
import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
import warnings
import numpy as np

# Suppress warnings and initialize resources
warnings.filterwarnings('ignore')
try:
    # Re-initialize NLTK resources in the app context
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
except Exception as e:
    st.error(f"Error initializing NLTK resources: {e}")
    sid = None

# --- Configuration ---
FILE_PATH = 'training.1600000.processed.noemoticon.csv'
COLUMN_NAMES = ['target', 'id', 'date', 'flag', 'user', 'text']
SAMPLE_SIZE = 100000 

# --- Functions for Processing and Classification ---

def preprocess_text(text, stop_words):
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

def get_vader_sentiment(text):
    if not sid: return 2
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 1  # Positive
    elif compound_score <= -0.05:
        return 0  # Negative
    else:
        return 2  # Neutral

def map_to_visual(sentiment_code):
    if sentiment_code == 1:
        return '🟢 Positive 😊'
    elif sentiment_code == 0:
        return '🔴 Negative 😡'
    else:
        return '🟡 Neutral 😐'

# --- Data Loading, Processing, and Caching (Streamlit optimization) ---

@st.cache_data
def load_and_classify_data():
    try:
        df = pd.read_csv(FILE_PATH, encoding='ISO-8859-1', names=COLUMN_NAMES)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {FILE_PATH}. Please ensure the file is in the current directory.")
        return pd.DataFrame()

    df = df.drop(columns=['id', 'date', 'flag', 'user'])
    df['target'] = df['target'].replace(4, 1) 

    # Take a balanced sample using stratification (if possible) or a large random sample
    # Using a large random sample to mitigate the previous imbalance issue
    sample_df = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    
    # Check for presence of both classes in the sample to prevent the Support=0 error
    if len(sample_df['target'].unique()) < 2:
        st.warning("The selected sample does not contain both sentiment classes. Model evaluation may fail.")

    sample_df['cleaned_text'] = sample_df['text'].apply(lambda x: preprocess_text(x, stop_words))
    sample_df['vader_prediction'] = sample_df['cleaned_text'].apply(get_vader_sentiment)
    sample_df['Visual_Result'] = sample_df['vader_prediction'].apply(map_to_visual)

    return sample_df

# --- Streamlit Web Application Interface ---

st.set_page_config(page_title="SA Lexicon Model", layout="wide")
st.title("Sentiment Analysis Project: Lexicon-Based Model (VADER)")
st.markdown("---")

df_results = load_and_classify_data()

if not df_results.empty:
    
    # 1. Interactive Test Interface
    st.header("1. Interactive Tweet Demo")
    user_input = st.text_area("Enter a tweet to classify:", placeholder="I am so happy with this service, it is amazing!")
    
    if user_input:
        cleaned_input = preprocess_text(user_input, stop_words)
        prediction_code = get_vader_sentiment(cleaned_input)
        visual_result = map_to_visual(prediction_code)
        
        st.subheader("Classification Result:")
        st.write(f"**Your Tweet:** *{user_input}*")
        # Use markdown for the emoji result for size and visibility
        st.markdown(f"**Prediction:** <div style='font-size: 28px;'>{visual_result}</div>", unsafe_allow_html=True)
        st.markdown("---")


    # 2. Performance Evaluation Results 
    st.header("2. Model Evaluation on Sample Data")
    
    # Filter out Neutral predictions for binary evaluation
    evaluation_df = df_results[df_results['vader_prediction'] != 2].copy()
    
    # Check if evaluation_df has both classes for comparison
    if len(evaluation_df['target'].unique()) < 2:
         st.error("Evaluation Error: The sample for evaluation is missing one or both original target classes. Evaluation report cannot be generated.")
    else:
        y_true = evaluation_df['target']
        y_pred = evaluation_df['vader_prediction']

        accuracy = accuracy_score(y_true, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records Analyzed", f"{len(evaluation_df)} Records")
        with col2:
            st.metric("Model Accuracy (Pos/Neg)", f"{accuracy:.4f}")

        st.subheader("Detailed Classification Report:")
        report = classification_report(y_true, y_pred, target_names=['Negative (0)', 'Positive (1)'], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        st.markdown("---")

    # 3. Sample Visualization 
    st.header("3. Random Sample Visualization")
    st.markdown("Showing 10 randomly selected tweets from the dataset sample with visual classification:")
    
    display_df = df_results[['text', 'Visual_Result']].sample(10)
    display_df.columns = ['Original Tweet', 'Predicted Sentiment']
    st.table(display_df)
