# IMDb Movie Review Sentiment Analysis

This project builds and evaluates several machine learning models to perform sentiment analysis on the Stanford IMDb movie review dataset. The goal is to classify a review as either positive (1) or negative (0).

The best-performing model was **Logistic Regression**, which achieved an **89% accuracy** on the test data.

## Dataset
* **Source:** Stanford [aclImdb_v1.tar.gz](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
* **Content:** 50,000 movie reviews.
* **Balance:** The dataset is perfectly balanced with 25,000 positive and 25,000 negative reviews.

## Methodology

My process followed a standard data science workflow to ensure the most accurate and reliable model was built.

### 1. Data Loading & Structuring
The raw data was downloaded (`!wget`) and extracted (`!tar`) from the `.tar.gz` archive. The individual text files from the `train/pos`, `train/neg`, `test/pos`, and `test/neg` directories were read and compiled into a single Pandas DataFrame with 'review' and 'label' columns.

### 2. Text Preprocessing
A comprehensive text cleaning pipeline was applied to each review:
1.  **Cleaning:** HTML tags, non-alphanumeric characters, and extra whitespace were removed. Text was converted to lowercase.
2.  **Stopword Removal:** Common English stopwords (e.g., "the", "is", "a") were removed using NLTK.
3.  **Tokenization:** Reviews were broken into individual words (tokens).
4.  **Lemmatization:** Words were reduced to their root form (e.g., "movies" became "movie") using NLTK's `WordNetLemmatizer`.
5.  **POS Tagging:** Part-of-Speech tags were applied to the lemmatized tokens.

### 3. Vectorization
The preprocessed text was converted into a numerical format using `TfidfVectorizer`, which scores words based on their frequency and importance across the entire dataset.

### 4. Model Training & Evaluation
The vectorized data was split into an 80% training set and a 20% test set. Four different classification models were trained and evaluated.

## Model Performance

The accuracy of each model on the unseen test data was as follows:

| Model | Test Accuracy |
| :--- | :--- |
| **Logistic Regression** | **89%** |
| XGBoost | 85% |
| Random Forest | 84% |
| K-Nearest Neighbors (KNN) | 78% |

Based on these results, **Logistic Regression** was the most effective model for this specific preprocessing pipeline.

## How to Use

### 1. Prerequisites
This project requires Python and several libraries. You can install them using pip:
```bash
pip install pandas nltk scikit-learn xgboost seaborn matplotlib
```
You will also need to download the NLTK resources used in the notebook. You can do this by running the following in Python:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

### 2. Running the Notebook
The notebook is designed to be run in an environment like Google Colab or a local Jupyter server.

The first cell (starting with !wget) will download and extract the aclImdb_v1.tar.gz dataset into the correct folder structure.

Run the cells sequentially to load, process, train, and evaluate the models.
