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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# importing dataset fromdrive
df=pd.read_csv("/content/drive/MyDrive/spam.csv",encoding='latin-1')

df
df.head()
df.isnull()
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df
df.isnull()
df.isnull().sum()
df.info()
df.describe()
df.duplicated()
df.drop_duplicates(inplace=True)
df
# mapping or replacing the values
# spam----> 1
# ham------> 0
mapping = {'spam': 1, 'ham': 0}
df['v1'] = df['v1'].map(mapping)

value_counts = df['v1'].value_counts()

# Creating a bar plot
plt.figure(figsize=(8, 6))
plt.bar(value_counts.index, value_counts.values)
plt.xlabel('v1')
plt.ylabel('Count')
plt.title('Distribution of v1')
plt.xticks(value_counts.index)
plt.show()

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    return text

df['v2'] = df['v2'].apply(preprocess_text)

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec= vectorizer.transform(X_test)

print(X_train_vec)

print(X_test_vec)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#ALSO USING THE SVM ALOGRITHM

from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train_vec, y_train)

svm_y_pred = svm_model.predict(X_test_vec)

svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f"SVM Accuracy: {svm_accuracy}")

print(confusion_matrix(y_test, svm_y_pred))
print(classification_report(y_test, svm_y_pred))
      
