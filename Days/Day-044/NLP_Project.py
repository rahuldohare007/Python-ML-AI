# 🚀 Day 44/100 of #100DaysOfCode
# 🎯 NLP Project: Movie Review Sentiment Analysis

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load Dataset (IMDB or any sentiment dataset)

data = pd.read_csv("movie_review.csv", encoding="latin-1")
df = pd.DataFrame(data)


# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # remove punctuation
    return text


df["cleaned"] = df["review"].astype(str).apply(clean_text)

# Train-Test Split
X = df["cleaned"]
y = df["tag"].map({"pos": 1, "neg": 0})

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Streamlit Frontend
st.set_page_config(page_title="🎬 Movie Review Sentiment Analysis", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Type a movie review below and see if it’s **Positive or Negative**!")

# Input box
user_input = st.text_area("Enter your review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]

        if prediction == "positive":
            st.success("✅ Positive Review – You’ll enjoy this movie!")
        else:
            st.error("❌ Negative Review – Probably not worth your time.")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Footer
st.markdown("---")
st.markdown("Built with 🖤 using **Scikit-learn + Streamlit**")
