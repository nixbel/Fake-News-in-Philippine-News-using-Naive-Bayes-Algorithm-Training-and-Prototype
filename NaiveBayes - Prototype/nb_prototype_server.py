from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import numpy as np

app = Flask(__name__)
CORS(app)

try:
    # Load the saved models
    tfidf_vectorizer = joblib.load(r'NB\tfidf_vectorizer.joblib')
    calibrated_model = joblib.load(r'NB\calibrated_naivebayes_model.joblib')

    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    
except Exception as e:
    print(f"Error loading models or NLTK data: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        
        # Preprocess text and get tokens
        processed_text, text_tokens = preprocess_text(text)
        
        # Vectorize the text
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        
        # Get prediction probability
        probability = calibrated_model.predict_proba(vectorized_text)[0]
        suspicious_probability = float(probability[1])
        
        # Update credibility logic based on confidence level
        if suspicious_probability <= 0.8:
            credibility = "Credible"
        elif suspicious_probability >= 0.8:
            credibility = "Suspicious"
        else:
            credibility = "Suspicious"
        
        # Get influential words
        credible_words, suspicious_words = get_influential_words(text_tokens, vectorized_text)
        
        # Generate summary
        summary = generate_summary(text)
        
        return jsonify({
            'credibility': credibility,
            'suspicious_probability': suspicious_probability,
            'summary': summary,
            'credible_words': credible_words,
            'suspicious_words': suspicious_words
        })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return ' '.join(lemmatized_tokens), lemmatized_tokens

def get_influential_words(text_tokens, vectorized_text, n_words=10):
    try:
        # Get feature names (words) from vectorizer
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get the base estimator from the calibrated model
        base_classifier = calibrated_model.calibrated_classifiers_[0].estimator
        
        # Get the coefficients from the Naive Bayes model
        coefficients = base_classifier.feature_log_prob_[1] - base_classifier.feature_log_prob_[0]
        
        # Get non-zero elements from vectorized text
        non_zero_indices = vectorized_text.nonzero()[1]
        
        # Create word-coefficient pairs
        words_coeffs = [(feature_names[i], coefficients[i]) for i in non_zero_indices if i < len(feature_names)]
        words_coeffs.sort(key=lambda x: x[1])
        
        # Get credible and suspicious words
        credible_words = [word for word, coeff in words_coeffs[:n_words] if word in text_tokens]
        suspicious_words = [word for word, coeff in words_coeffs[-n_words:] if word in text_tokens]
        
        return credible_words, suspicious_words
    except Exception as e:
        print(f"Error in get_influential_words: {str(e)}")
        return [], []

def generate_summary(text, sentences_count=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")
        return text[:200] + "..."  # Return truncated text as fallback

if __name__ == '__main__':
    app.run(debug=True)