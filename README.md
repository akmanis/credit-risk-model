# Credit Risk Prediction System

A machine learning system that predicts the probability of loan default using the **German Credit Dataset**.

The system analyzes borrower financial information and estimates the **risk of credit default**, helping simulate how banks and fintech companies perform **credit risk assessment**.

---

## Features

* Logistic Regression and Random Forest models
* Default probability prediction
* Risk score (0–100)
* Loan approval decision
* Feature importance visualization
* Confusion matrix and ROC curve
* Interactive dashboard using Streamlit

---

## Application Interface

### Credit Risk Prediction Dashboard
```

```

This interface allows users to enter loan applicant details such as:

* Credit Amount
* Loan Duration
* Age
* Account Status Score
* Credit History Score

The model then predicts the probability of loan default.

---

### Model Confusion Matrix

Insert screenshot here:

```
images/confusion_matrix.png
```

The confusion matrix shows how well the model classifies borrowers into:

* No Default
* Default

It displays True Positives, True Negatives, False Positives, and False Negatives.

---

### ROC Curve

Insert screenshot here:

```
images/roc_curve.png
```

The ROC curve measures the model’s ability to distinguish between safe and risky borrowers.

The **AUC score** indicates the overall model performance.

---

### Feature Importance

Insert screenshot here:

```
images/feature_importance.png
```

This chart shows which financial variables most influence the model's prediction.

Important features include:

* Credit Amount
* Account Status
* Loan Duration
* Age
* Credit History

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

## Dataset

German Credit Dataset

The dataset contains **21 financial features** describing borrower profiles such as:

* Credit amount
* Loan duration
* Account status
* Credit history
* Employment years
* Housing status
* Number of existing credits
* Age
* Job status

These features help estimate the probability that a borrower may **default on a loan**.

---

## How to Run

Clone the repository:

```
git clone https://github.com/akmanish/credit-risk-model.git
cd credit-risk-model
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app/app.py
```

---

## Project Structure

```
credit-risk-model
│
├── app/
│   └── app.py              → Streamlit dashboard
│
├── data/
│   └── german_credit_data.csv
│
├── model/
│   └── train_model.py      → Model training script
│
├── credit_model.pkl        → Trained machine learning model
├── requirements.txt
└── README.md
```

---

## Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 74%      |
| Random Forest       | 82%      |

Random Forest was selected as the final model due to higher predictive accuracy.

---

## Author

Manish AK
BS Economic Sciences
Indian Institute of Science Education and Research (IISER) Bhopal

Interested in:

* Fintech
* Machine Learning
* Credit Risk Modeling
* Financial Data Science
