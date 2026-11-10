# OIBSIP-DataAnalystics-L4-Sentiment-Analysis-
# Task 4: Sentiment Analysis Using Machine Learning

## Project Overview

This project implements a sentiment analysis system that classifies text into three categories: Positive, Negative, and Neutral. It uses natural language processing (NLP) and machine learning techniques to analyse text and understand opinions expressed in customer feedback and social media posts.

## Objectives

- Load and explore a sentiment analysis dataset.
- Preprocess text by converting it to lowercase, removing punctuation and URLs, tokenising, and removing selected stopwords.
- Convert text into numerical features using TF-IDF.
- Split the dataset into 80% training data and 20% testing data.
- Train two classifiers: Multinomial Naive Bayes and Logistic Regression.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Visualise sentiment distribution and generate WordClouds for each sentiment class.
- Analyse five misclassified examples.
- Compare model performance and identify the best-performing classifier.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK stopword resources through scikit-learn's built-in English stopword list
- Matplotlib
- Seaborn
- WordCloud
- Jupyter Notebook

## Dataset

**Dataset:** Twitter US Airline Sentiment

**Source:** https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

The dataset contains tweets about US airlines labelled as positive, negative, or neutral. The project uses the tweet text and sentiment label to train and evaluate the classifiers.

The primary input file is `Tweets.csv`.

## Methodology

### 1. Data Loading and Exploration

Load the dataset using Pandas, inspect the columns, identify missing values and duplicate records, and examine the distribution of sentiment labels.

### 2. Text Preprocessing

Clean the tweet text by converting it to lowercase, removing URLs and user mentions, normalising punctuation, tokenising, and removing selected stopwords. Important negation words are preserved because they can change the meaning of a sentence.

### 3. Feature Extraction

TF-IDF converts text into numerical features that machine learning algorithms can process. It gives greater relative importance to informative words and less weight to terms that occur across many documents.

### 4. Model Training

Train two classification algorithms:

- **Multinomial Naive Bayes:** A fast baseline commonly used for text classification.
- **Logistic Regression:** A linear classifier that works effectively with sparse TF-IDF features.

The dataset is divided into 80% training data and 20% testing data.

### 5. Model Evaluation

Evaluate both classifiers using accuracy, macro precision, macro recall, macro F1-score, weighted F1-score, and confusion matrices.

### 6. Visualisation

Generate a bar chart showing the distribution of positive, negative, and neutral tweets. Create WordClouds for each sentiment class and performance comparison charts for the classifiers.

### 7. Error Analysis

Inspect five misclassified examples from the best-performing model. Discuss possible causes, including negation, sarcasm, ambiguous text, mixed opinions, and insufficient context.

### 8. Model Saving

Save the best-performing trained pipeline using Joblib so it can be loaded later for sentiment predictions.

## Project Structure

```text
Task4-Sentiment-Analysis/
│
├── Tweets.csv
├── Task4_Sentiment_Analysis.ipynb
├── README.md
│
└── output/
    ├── sentiment_distribution.png
    ├── sentiment_wordclouds.png
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── confusion_matrix_naive_bayes.png
    ├── confusion_matrix_logistic_regression.png
    ├── five_misclassified_examples.csv
    └── sentiment_model.joblib
```

## Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud joblib
```

## How to Run

1. Download `Tweets.csv` from the dataset source.
2. Place the CSV file in the project directory.
3. Open a terminal and launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `Task4_Sentiment_Analysis.ipynb`.
5. Execute all cells in order.
6. Review the evaluation metrics, confusion matrices, WordClouds, and error analysis.
7. Check the `output/` directory for generated files.

## Expected Outcomes

- A trained three-class sentiment classification system.
- A comparison of Naive Bayes and Logistic Regression.
- Evaluation metrics and confusion matrices for both models.
- Visualisations of sentiment distribution and word frequencies.
- Five examples of misclassified tweets with investigation notes.
- A saved model pipeline for future predictions.

The actual model scores depend on the dataset, preprocessing choices, and training configuration. They should be reported from the notebook's evaluation results rather than assumed in advance.

## Applications

- Customer feedback analysis.
- Social media monitoring.
- Airline service quality assessment.
- Brand reputation tracking.
- Public opinion analysis.

## Conclusion

This project demonstrates how natural language processing and machine learning can be used to classify text into positive, negative, and neutral sentiments. TF-IDF provides numerical text features, while Naive Bayes and Logistic Regression learn patterns associated with sentiment labels. Model evaluation, visualisation, and error analysis help assess the system's strengths and limitations.

## Author
Natarajan Surya
**Project:** Task 4 — Sentiment Analysis  
**Domain:** Natural Language Processing and Machine Learning  
**Tools:** Python, Pandas, Scikit-learn, Jupyter Notebook
