# Sentiment Analysis Using Machine Learning and Word Embeddings

## Overview

This project focuses on performing sentiment analysis on Twitter data using Natural Language Processing (NLP), Traditional Machine Learning, Word Embeddings, and Deep Learning techniques.

The main objective is to classify tweets into two sentiment categories: Positive and Negative.

The project compares traditional machine learning algorithms using TF-IDF features with deep learning models that use word embeddings learned from scratch through CBOW and Skip-gram architectures.

## Dataset

**Sentiment140 Dataset**

- Source: [Kaggle – Sentiment140](https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv)
- Total Samples: 1,600,000 tweets.
- Original Labels:
  - 0: Negative
  - 4: Positive

The labels are converted into binary classification format:

- 0: Negative
- 1: Positive

## Project Workflow

### 1. Data Loading and Preprocessing

The dataset is loaded and prepared for sentiment classification.

Preprocessing operations include:

- Removing missing and duplicate values.
- Converting text to lowercase.
- Removing URLs and user mentions.
- Removing HTML tags.
- Removing unnecessary special characters.
- Removing extra whitespace.
- Converting sentiment labels into binary values.

Stopwords that may affect sentiment meaning, such as negation words, are preserved.

### 2. TF-IDF Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert tweets into numerical feature vectors.

Configuration:

- Maximum Features: 100,000
- N-gram Range: (1, 2)
- Minimum Document Frequency: 2

The vectorizer is fitted on training data only to prevent data leakage.

### 3. Traditional Machine Learning Models

Three machine learning algorithms are implemented using TF-IDF features:

**Bernoulli Naive Bayes**
- A probabilistic classifier based on binary feature occurrence.

**Linear Support Vector Machine (SVM)**
- A linear classification algorithm that finds an optimal decision boundary between sentiment classes.

**Logistic Regression**
- A classification algorithm that estimates the probability of a tweet belonging to a sentiment category.

### 4. CBOW Word Embeddings

CBOW (Continuous Bag of Words) is implemented and trained from scratch using training tweets.

The model learns word representations by predicting a target word from its surrounding context words.

The learned embedding vectors are extracted and used as input representations for a deep learning sentiment classifier.

### 5. Skip-gram Word Embeddings

Skip-gram is implemented and trained from scratch without using pretrained embeddings.

Unlike CBOW, Skip-gram predicts surrounding context words from a target word.

The learned embeddings are extracted and used in a separate deep learning classification model.

### 6. Deep Learning Classification

Two deep learning sentiment classification models are developed:

**CBOW + Deep Learning**
- Uses embeddings learned through CBOW.
- Global Average Pooling layer.
- Dense hidden layer with ReLU activation.
- Dropout regularization.
- Sigmoid output layer.

**Skip-gram + Deep Learning**
- Uses embeddings learned through Skip-gram.
- Global Average Pooling layer.
- Dense hidden layer with ReLU activation.
- Dropout regularization.
- Sigmoid output layer.

The embedding weights are initialized using the embeddings learned from scratch and fine-tuned during classification training.

## Models Comparison

The project evaluates five models:

1. Bernoulli Naive Bayes
2. Linear SVM
3. Logistic Regression
4. CBOW + Deep Learning
5. Skip-gram + Deep Learning

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

A final comparison table is generated to identify the best-performing model.

## Sample Predictions

The trained models are tested using the following example tweets:

- "I love this!"
- "I hate that!"
- "It was okay, not great."

Each model predicts whether the sentiment is Positive or Negative.

## Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Natural Language Processing (NLP)

## Project Structure

```text
Sentiment-Analysis/
│
├── Sentiment_Analysis.ipynb
├── training.1600000.processed.noemoticon.csv
│
├── cbow_embeddings.npy
├── skipgram_embeddings.npy
├── cbow_love_vector.npy
├── skipgram_love_vector.npy
│
├── model_comparison.csv
├── sample_predictions.csv
│
├── bernoulli_nb.pkl
├── linear_svm.pkl
├── logistic_regression.pkl
├── tfidf_vectorizer.pkl
│
├── cbow_model.keras
├── skipgram_model.keras
├── cbow_classifier.keras
└── skipgram_classifier.keras
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/USERNAME/Sentiment-Analysis-Using-Machine-Learning-and-Word-Embeddings.git
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### 3. Download the Dataset

Download the Sentiment140 dataset from Kaggle and place the CSV file in the project directory.

### 4. Run the Notebook

```bash
jupyter notebook
```

Open `Sentiment_Analysis.ipynb` and execute the cells sequentially.

## Key Learning Outcomes

- Understanding NLP preprocessing techniques.
- Applying TF-IDF for text representation.
- Implementing traditional machine learning classifiers.
- Understanding and implementing CBOW and Skip-gram.
- Learning word embeddings from scratch.
- Integrating word embeddings with deep learning models.
- Evaluating and comparing text classification models.

## Conclusion

This project provides a comparative study of traditional machine learning and deep learning approaches for Twitter sentiment analysis.

It demonstrates how different text representation techniques and model architectures influence sentiment classification performance.

The final results are used to identify the most effective approach for binary sentiment classification.

---

**Project Category:** Natural Language Processing (NLP) | Machine Learning | Deep Learning

**Dataset:** Sentiment140
