# 🚀 Day 30/100 of #100DaysOfCode
# 🎯 Project: Spam vs Ham Classifier 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("../data/SMSSpamCollection", sep='\t', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Encode labels (ham=0, spam=1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Basic EDA
print("Dataset shape:", df.shape)
print(df['label'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Distribution of Spam vs Ham")
plt.show()

# 4. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

df['clean_message'] = df['message'].apply(clean_text)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

# 6. Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 7. Train Models
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)


# 8. Evaluate Models
for name, model in [('Naive Bayes', nb_model), ('Logistic Regression', lr_model)]:
    y_pred = model.predict(X_test_tfidf)
    print(f"\n=== {name} Performance ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 9. Save Best Model (Naive Bayes)
joblib.dump(nb_model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
