# Spam vs Ham Classifier

This project implements a **Spam vs Ham Email/SMS Classifier** using **Python, Scikit-learn, and NLP techniques**.  
It classifies text messages as **spam** (unwanted) or **ham** (legitimate).

---

## **Features**
- Preprocessing of raw text (lowercasing, punctuation removal, stopword removal)
- Feature extraction using **TF-IDF Vectorizer**
- Model training using **Multinomial Naive Bayes** and **Logistic Regression**
- Evaluation using accuracy, confusion matrix, and classification report
- Saves trained model and vectorizer using `joblib` for future use

---

## **Tech Stack**
- **Python 3.x**
- **Pandas, Numpy, Matplotlib, Seaborn** (data handling and visualization)
- **Scikit-learn** (model training and evaluation)
- **Joblib** (model persistence)

---

## **Dataset**
We use the **SMS Spam Collection Dataset** from UCI Machine Learning Repository:  
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection  

- **Columns:**
  - `label` → spam or ham  
  - `message` → text content  

---