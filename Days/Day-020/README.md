# Titanic Survival Prediction – Logistic Regression

## 📌 Overview
This project uses **Logistic Regression** to predict whether a passenger survived the Titanic disaster based on demographic and travel information.  
It includes **feature engineering**, **data preprocessing**, and **model evaluation** with multiple performance metrics.

---

## 📂 Dataset
The dataset contains information about Titanic passengers such as:
- Passenger class (`Pclass`)
- Name, Sex, and Age
- Number of siblings/spouses aboard (`SibSp`)
- Number of parents/children aboard (`Parch`)
- Ticket fare (`Fare`)
- Port of embarkation (`Embarked`)
- Survival status (`Survived`) – **Target variable**

📌 *Source:* [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)

---

## ⚙️ Features & Preprocessing
- **Extracted Titles** from passenger names (`Mr`, `Mrs`, `Miss`, etc.)
- **Mapped gender** (`male` → 0, `female` → 1)
- **Filled missing values** (Age, Fare, Embarked)
- **Binned Age and Fare** into categories to reduce outlier effect
- **Created new features**:
  - `FamilySize` = `SibSp` + `Parch` + 1
  - `IsAlone` → whether the passenger was alone or not
- **One-hot encoding** for categorical variables
- **Feature scaling** with `StandardScaler`

---

## 🛠️ Tech Stack
- **Python 3**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Machine learning model & evaluation

---

## 📊 Model & Evaluation
Algorithm used: **Logistic Regression**

### Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Score
- 5-Fold Cross-Validation

---

## 📈 Results
- **Confusion Matrix** for classification analysis
- **ROC Curve** for performance visualization
- **Cross-validation** to check model stability

---

