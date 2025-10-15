
📧 Spam Message Detection using Machine Learning

This project implements a Spam Message Detection System using Natural Language Processing (NLP) and Machine Learning techniques.
The model classifies text messages as Spam (1) or Ham (0) based on their content using Logistic Regression and Support Vector Machine (SVM) algorithms.

🚀 Project Overview

The goal of this project is to automatically detect whether a given SMS message is spam or not.
The dataset used is the SMS Spam Collection Dataset, which contains labeled messages as either "spam" or "ham."

🧠 Key Features

Text preprocessing using Regular Expressions, Stopword Removal, and Lemmatization

TF-IDF Vectorization for feature extraction

Classification using:

Logistic Regression

Support Vector Machine (SVM)

Evaluation with:

Accuracy

Confusion Matrix

Classification Report


Source: Commonly available SMS Spam Collection Dataset (UCI Machine Learning Repository or Kaggle)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

⚙️ Steps in the Pipeline
1️⃣ Data Cleaning

Removed unwanted columns (Unnamed: 2, 3, 4)

Dropped duplicates

Mapped labels:

spam → 1

ham → 0

2️⃣ Text Preprocessing

Removed non-alphabetic characters

Converted text to lowercase

Removed stopwords

Lemmatized words

Joined tokens back into clean sentences

##Model Training and Evaluation

Logistic Regression

SVM (Support Vector Machine)
















