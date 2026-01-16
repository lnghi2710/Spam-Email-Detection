# Spam Email Classification

## Introduction
This project aims to build a machine learning model to classify emails as **Spam** or **Ham (Not Spam)**.  
It demonstrates a complete NLP workflow including data preprocessing, feature extraction, model training, and evaluation using Logistic Regression.

---

## Features
- Text cleaning and normalization
- TF-IDF feature extraction with n-grams
- Spam vs Ham email classification
- Model evaluation with standard classification metrics
- Simple, interpretable, and efficient machine learning approach

---

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

Required libraries:
- numpy
- pandas
- scikit-learn
- matplotlib

---

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/lnghi2710/Spam-Email-Detection.git
```
```bash
pip install -r requirements.txt
```

## Usage
To run the project, execute the main training script:

```bash
python data_chart.py
```

The script will automatically:
- Analyze the distribution of Spam and Ham emails
- Visualize Spam vs Ham distribution using a bar chart with value annotations
- Compute the length of each email based on character count
- Visualize email text length distribution using a histogram

```bash
python model.py
```

The script will automatically:
- Load and preprocess the email dataset
- Convert text data into TF-IDF features
- Train the Logistic Regression model
- Evaluate the model and display performance metrics


## Dataset

The dataset consists of labeled email messages classified into two categories: Spam and Ham.

Main columns:
- text: raw email content
- label: class name (spam or ham)
- label_num: numeric label (0 = ham, 1 = spam)

Dataset Issues:
- Imbalanced class distribution between spam and ham emails
- Presence of noisy characters such as numbers, symbols, and mixed casing
  
Solutions Applied:
- Text normalization and cleaning using regular expressions
- TF-IDF vectorization to handle sparse text data
- Class weighting during model training to reduce bias

## Model Training
- Algorithm: Logistic Regression
- Feature extraction: TF-IDF with unigrams and bigrams
- Train-test split: 80% training, 20% testing
- Class weights applied to improve spam detection performance
Logistic Regression is chosen because it performs well on high-dimensional sparse data and provides interpretable results.

## Evaluation
The model performance is evaluated using:
- Accuracy score
- Precision, Recall, and F1-score
- Confusion matrix
These metrics provide insight into how effectively the model distinguishes spam emails from legitimate ones.

## Contributing
- Contributions are welcome.
- Please fork the repository, create a new branch, and submit a pull request for review.
