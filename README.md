### Fake News Detection in Philippine News using Naive Bayes Algorithm

> Developed by **Team LiveANet**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Machine Learning Model](#machine-learning-model)
- [Installation & Setup](#installation--setup)
  - [Backend Setup](#backend-setup)
  - [Chrome Extension Setup](#chrome-extension-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Known Issues & Limitations](#known-issues--limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**E For Real** is a browser extension and machine learning-powered tool designed to detect potentially fake or misleading news articles — with a specific focus on **Philippine news content**. It uses a **Calibrated Naive Bayes classifier** trained on a labeled Filipino news dataset, combined with **TF-IDF vectorization**, to analyze the text of any webpage and return a credibility assessment in real time.

The system consists of two major components:
1. A **Flask backend server** that handles text preprocessing, model inference, word influence analysis, and summarization.
2. A **Chrome Extension (Manifest V3)** that extracts page content, communicates with the backend, and visually presents the analysis results to the user.

---

## Features

- ✅ **Real-time credibility analysis** of news articles directly in the browser
- 📊 **Suspicious probability score** with animated progress bar (color-coded from green to red)
- 📝 **Automatic summarization** of article content using LSA (Latent Semantic Analysis)
- 🔍 **Influential word highlighting** — marks credible (green) and suspicious (red) words on the page
- 💡 **Fake news detection tips** shown after every analysis
- ⚡ Lightweight extension with minimal permissions

---

## Project Structure

```
Fake-News-in-Philippine-News-using-Naive-Bayes-Algorithm-Training-and-Prototype/
│
├── NB/
│   ├── calibrated_naivebayes_model.joblib   # Trained & calibrated Naive Bayes model
│   └── tfidf_vectorizer.joblib              # Fitted TF-IDF vectorizer
│
├── extension/
│   ├── manifest.json                        # Chrome Extension manifest (V3)
│   ├── popup.html                           # Extension popup UI
│   ├── popup.js                             # Popup logic & API calls
│   ├── content.js                           # Page content extraction & word highlighting
│   └── img/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
│
├── nb_prototype_server.py                   # Flask backend server
├── Train_NaiveBayes.ipynb                   # Model training notebook
└── README.md
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Chrome Browser                     │
│                                                      │
│  ┌──────────────┐         ┌──────────────────────┐  │
│  │  popup.html  │◄───────►│    popup.js           │  │
│  │  (UI Layer)  │         │  (Logic & API Calls)  │  │
│  └──────────────┘         └──────────┬───────────┘  │
│                                       │               │
│                            chrome.tabs.sendMessage    │
│                                       │               │
│                           ┌───────────▼────────────┐ │
│                           │      content.js         │ │
│                           │  (Text Extraction +     │ │
│                           │   Word Highlighting)    │ │
│                           └───────────┬────────────┘ │
└───────────────────────────────────────┼──────────────┘
                                        │ HTTP POST
                                        ▼
                         ┌──────────────────────────────┐
                         │    Flask Backend (port 5000)  │
                         │   nb_prototype_server.py      │
                         │                              │
                         │  1. Text Preprocessing       │
                         │  2. TF-IDF Vectorization     │
                         │  3. Naive Bayes Prediction   │
                         │  4. Word Influence Analysis  │
                         │  5. LSA Summarization        │
                         └──────────────────────────────┘
```

---

## Machine Learning Model

### Algorithm: Calibrated Multinomial Naive Bayes

| Component | Details |
|-----------|---------|
| Base Classifier | Multinomial Naive Bayes |
| Calibration | `CalibratedClassifierCV` (sigmoid or isotonic) |
| Vectorization | TF-IDF (`TfidfVectorizer`) |
| Output | Probability score: `[credible_prob, suspicious_prob]` |
| Decision Threshold | `suspicious_probability >= 0.80` → Suspicious |

### Why Naive Bayes?

Naive Bayes is well-suited for text classification tasks due to:
- Efficiency on high-dimensional sparse data (TF-IDF vectors)
- Strong baseline performance with limited training data
- Probabilistic output that enables credibility confidence scoring
- Interpretability — feature log-probabilities can be used to identify influential words

### Calibration

The model is wrapped with `CalibratedClassifierCV` to improve the reliability of probability estimates. Raw Naive Bayes probabilities tend to be overconfident, so calibration maps them to more realistic values.

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Google Chrome (for the extension)
- pip

---

### Backend Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Fake-News-in-Philippine-News-using-Naive-Bayes-Algorithm-Training-and-Prototype.git
cd Fake-News-in-Philippine-News-using-Naive-Bayes-Algorithm-Training-and-Prototype
```

#### 2. Install Python Dependencies

```bash
pip install flask flask-cors joblib nltk sumy numpy scikit-learn
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt` (recommended):**
```
flask==3.0.0
flask-cors==4.0.0
joblib==1.3.2
nltk==3.8.1
sumy==0.11.0
numpy==1.26.0
scikit-learn==1.4.0
```

#### 3. Download NLTK Data

The server downloads the required NLTK resources automatically on startup. Alternatively, pre-download them manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

#### 4. Ensure Model Files Are in Place

Make sure the following files are inside a folder named `NB/` at the same level as the server script:

```
NB/
├── calibrated_naivebayes_model.joblib
└── tfidf_vectorizer.joblib
```

#### 5. Start the Server

```bash
python nb_prototype_server.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

The backend is now live and listening at `http://localhost:5000`.

---

### Chrome Extension Setup

#### 1. Open Chrome Extensions

Navigate to `chrome://extensions/` in your browser.

#### 2. Enable Developer Mode

Toggle **Developer mode** on (top-right corner).

#### 3. Load the Extension

Click **"Load unpacked"** and select the `extension/` folder (the one containing `manifest.json`).

#### 4. Verify Installation

The **E For Real** icon should appear in your Chrome toolbar. Pin it for easy access.

---

## Usage

1. **Navigate** to any Philippine news article or webpage.
2. **Click** the E For Real extension icon in your toolbar.
3. **Click** the **"Analyze Content"** button.
4. Wait for the analysis to complete (a spinner is shown during processing).
5. View the results:
   - **Credibility label**: `Credible` (green) or `Suspicious` (red)
   - **Suspicious probability %** with an animated, color-coded progress bar
   - **Article summary** (highlighted based on credibility)
   - **Word influence panel**: credible words vs. suspicious words
   - **Fake news detection tips**
6. Highlighted words appear directly on the webpage — green for credible-leaning words, red for suspicious-leaning words.

---

## API Reference

### `POST /predict`

Analyzes the provided text and returns a credibility assessment.

**Request:**

```json
{
  "text": "Full article text to analyze..."
}
```

**Response:**

```json
{
  "credibility": "Credible" | "Suspicious",
  "suspicious_probability": 0.0 - 1.0,
  "summary": "LSA-generated summary of the article.",
  "credible_words": ["word1", "word2", ...],
  "suspicious_words": ["word3", "word4", ...]
}
```

**Error Response:**

```json
{
  "error": "Description of error"
}
```

**Example cURL call:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The government announced new policies today regarding infrastructure spending."}'
```

---

## How It Works

### 1. Text Extraction (`content.js`)

The extension's content script extracts the main readable text from the current page by prioritizing:
1. `<article>` tags
2. `<main>` tags
3. Full `document.body` as fallback

### 2. Text Preprocessing (`nb_prototype_server.py`)

The extracted text undergoes the following NLP pipeline:

```
Raw Text
   ↓
Lowercase conversion
   ↓
Remove special characters & digits (regex)
   ↓
Tokenization (NLTK word_tokenize)
   ↓
Stopword removal (NLTK English stopwords)
   ↓
Short word removal (len ≤ 2)
   ↓
Lemmatization (WordNetLemmatizer)
   ↓
Processed Token List
```

### 3. Vectorization

The preprocessed tokens are joined and transformed using the pre-fitted `TfidfVectorizer`, converting the text into a sparse numerical feature vector.

### 4. Prediction

The calibrated Naive Bayes model returns a probability array `[P(credible), P(suspicious)]`. If `P(suspicious) >= 0.80`, the article is classified as **Suspicious**; otherwise, it is **Credible**.

### 5. Influential Word Analysis

The model's internal feature log-probabilities are used to rank words:
- Words with **high log-probability difference** (suspicious - credible) → suspicious words
- Words with **low log-probability difference** (credible - suspicious) → credible words

Only words present in the article are returned.

### 6. Summarization

The `sumy` library's **LSA (Latent Semantic Analysis) Summarizer** generates a 3-sentence extractive summary of the article.

### 7. Results Display (`popup.js` + `content.js`)

Results are rendered in the popup UI and sent back to `content.js` for in-page word highlighting via `chrome.tabs.sendMessage`.

---

## Dataset

The model was trained on a labeled dataset of **Philippine news articles**, categorized as:
- `0` — Credible / Real News
- `1` — Suspicious / Fake News

> **Note:** Refer to `Train_NaiveBayes.ipynb` for dataset preprocessing steps, class distribution, and feature engineering details.

---

## Model Training

Open and run `Train_NaiveBayes.ipynb` to reproduce the training process. The notebook covers:

1. Dataset loading and exploration
2. Text preprocessing
3. TF-IDF feature extraction
4. Train/test split
5. Multinomial Naive Bayes training
6. Probability calibration with `CalibratedClassifierCV`
7. Evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)
8. Saving models with `joblib`

To retrain:
```bash
jupyter notebook Train_NaiveBayes.ipynb
```

---

## Known Issues & Limitations

| Issue | Description |
|-------|-------------|
| **Localhost dependency** | The extension requires the Flask server to be running locally on port 5000. It cannot work standalone. |
| **English-language bias** | NLTK stopwords and lemmatization are English-only. Filipino/Tagalog articles may not preprocess optimally. |
| **Threshold sensitivity** | The 0.80 threshold for "Suspicious" was tuned empirically and may not generalize to all article types. |
| **Short articles** | Very short texts (fewer than ~50 words) may yield unreliable predictions. |
| **Summarization fallback** | If the LSA summarizer fails (e.g., on very short text), it falls back to the first 200 characters. |
| **Mixed-language content** | Code-switching (Taglish) in Filipino articles may reduce model accuracy. |
| **No background script** | The extension uses no background service worker; results are not cached between popup sessions. |

---

## Future Improvements

- [ ] **Multilingual support** — Add Filipino/Tagalog stopwords and a Tagalog lemmatizer or stemmer
- [ ] **Cloud deployment** — Host the Flask backend on a cloud platform (e.g., Render, Railway, or GCP) to eliminate the localhost requirement
- [ ] **Model upgrade** — Experiment with transformer-based models (e.g., mBERT, RoBERTa) fine-tuned on Philippine news
- [ ] **Offline inference** — Run the model directly in the browser using ONNX or TensorFlow.js
- [ ] **Dataset expansion** — Collect more recent Filipino fake news samples to improve generalization
- [ ] **URL credibility scoring** — Integrate domain reputation databases as additional features
- [ ] **User feedback loop** — Allow users to flag incorrect predictions to improve the model over time
- [ ] **Firefox/Edge support** — Adapt the extension manifest for other Chromium-based browsers

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure your code is well-commented and that any changes to the model are documented in the notebook.

---

## License

This project is developed for academic purposes by **Team LiveANet**. All rights reserved.

---

## Author

Jacques Nico Belmonte - Programmer
John Louie Abenir - Project Leader
Kenn Louise Comprado - Technical Writer

> **Disclaimer:** This tool is intended to assist users in critically evaluating online news content. It is not a definitive arbiter of truth. Always cross-reference information with multiple credible sources.
