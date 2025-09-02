# 🚀 Day 41/100 of #100DaysOfCode
# 🎯 Intro to NLP + Text Preprocessing  

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sample text
text = "Natural Language Processing (NLP) is transforming AI! It's amazing, isn't it?"

# Step 1: Lowercasing
text = text.lower()

# Step 2: Remove punctuation and non-alphabetic characters
text = re.sub(r'[^a-z\s]', '', text)

# Step 3: Tokenization
tokens = word_tokenize(text)

# Step 4: Stopword removal
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Step 5: Stemming vs Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stems = [stemmer.stem(word) for word in tokens]
lemmas = [lemmatizer.lemmatize(word) for word in tokens]

print("Original Tokens:", tokens)
print("After Stemming:", stems)
print("After Lemmatization:", lemmas)
