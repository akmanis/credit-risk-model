# Credit Risk Prediction System

A machine learning system that predicts the probability of loan default using the German Credit Dataset.

## Features

* Logistic Regression and Random Forest models
* Default probability prediction
* Risk score (0–100)
* Loan approval decision
* Feature importance visualization
* Confusion matrix and ROC curve
* Interactive dashboard using Streamlit

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib

## Dataset

German Credit Dataset (21 financial features)

## How to Run

Clone the repository:

```
git clone https://github.com/akmanis/credit-risk-model.git
cd credit-risk-model
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app/app.py
```

## Project Structure

```
app/        → Streamlit dashboard  
data/       → Dataset  
model/      → Training script  
credit_model.pkl → Trained model  
```
