# Loan Default Prediction

A simple machine learning project that predicts whether a loan application will
be approved or not, based on applicant details such as income, credit history,
education, and property area.

## Project Overview

This project covers the full workflow of a classification problem:

- Exploratory Data Analysis (EDA) and visualization
- Data cleaning and missing value handling
- Encoding categorical features
- Feature scaling
- Training and tuning multiple ML models
- Model evaluation and comparison
- Exporting the best model for later use in a Streamlit web app

## Project Structure

```
loan default prediction/
|-- .streamlit/
|   |-- config.toml               Light theme configuration (green palette)
|
|-- data/
|   |-- loan_dataset.csv          Raw dataset
|   |-- processed_loan_data.csv   Cleaned and encoded dataset
|
|-- logs/
|   |-- app.log                   Text log of every app event
|   |-- predictions_log.csv       Structured log of every prediction made
|   |-- contact_messages.csv      Messages submitted through the About Us contact form
|
|-- models/
|   |-- best_model.pkl            Trained model with the best test accuracy
|   |-- scaler.pkl                StandardScaler fitted on training data
|   |-- encoders.pkl              LabelEncoders for categorical columns and target
|
|-- notebook/
|   |-- loan_default_prediction.ipynb   Main notebook with the full workflow
|
|-- styles/
|   |-- style.css                 All custom CSS for the Streamlit app
|
|-- app.py                        Streamlit web app for loan status prediction
|-- requirements.txt
|-- README.md
```

## Dataset

The dataset contains loan application records with the following columns:

| Column | Description |
|---|---|
| Loan_ID | Unique identifier for each application (dropped before training) |
| Gender | Male / Female |
| Married | Applicant marital status |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Income of the applicant |
| CoapplicantIncome | Income of the co-applicant |
| LoanAmount | Loan amount requested (in thousands) |
| Loan_Amount_Term | Term of the loan in months |
| Credit_History | Whether the applicant has a credit history (1) or not (0) |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target column: Y (approved) or N (not approved) |

## Setup

1. Clone or download this project folder.
2. Create a virtual environment (optional but recommended).

```
python -m venv .venv
.venv\Scripts\activate      (Windows)
source .venv/bin/activate   (Mac / Linux)
```

3. Install the required libraries.

```
pip install -r requirements.txt
```

4. Open the notebook.

```
jupyter notebook notebook/loan_default_prediction.ipynb
```

## Workflow Followed in the Notebook

1. Import libraries
2. Load the dataset
3. Exploratory Data Analysis and visualization
4. Check and handle missing values (mode for categorical, median for numerical)
5. Drop irrelevant columns
6. Encode categorical features
7. Save the processed dataset
8. Split dependent and independent features
9. Train test split
10. Feature scaling
11. Train models with GridSearchCV (Logistic Regression, Random Forest, SVM, KNN, XGBoost)
12. Compare model results
13. Select the best model
14. Export the best model, scaler, and encoders
15. Test the best model with custom input

## Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

Each model is tuned using `GridSearchCV` with 5-fold cross validation, and
compared on both training and test accuracy to check for overfitting.

## Output Files

After running the notebook, the following files are created inside the
`models` folder:

- `best_model.pkl` : the model with the highest test accuracy
- `scaler.pkl` : the StandardScaler fitted on the training data
- `encoders.pkl` : a dictionary of LabelEncoders for each categorical column
  and the target column

These three files are all that is needed to make predictions on new data,
for example inside a Streamlit web app.

## Testing with Custom Input

The last section of the notebook allows testing the trained model with a
custom applicant record. Update the values in the `custom_input` dictionary
and run the remaining cells to see whether the loan would be approved or not.

## Streamlit Web App

A Streamlit app (`app.py`) is included with a light theme that keeps the same
green color palette used throughout the project, and a floating bottom
navigation bar (visible on both desktop and mobile) with three pages:
Dashboard, Prediction, and About Us.

**Dashboard** (default page on launch)
Opens with a welcome header, followed by an overview of all predictions made
(total, approved, not approved, approval rate), a list of the 10 most recent
predictions with the input details used, and interactive charts built from
the prediction history:

- Approved vs Not Approved split
- Credit History vs Prediction outcome
- Applicant Income distribution
- Approval rate by Property Area

Charts are built with Plotly, so they support hovering for exact values,
dragging to zoom, and clicking legend entries to toggle a series on or off.

**Prediction**
A focused form with only the applicant detail fields and the prediction
output. All fields start empty ("Choose an option" for dropdowns, blank
placeholders for number fields) rather than pre-filled defaults, so a
submission always reflects values the user actually entered. Applicant and
coapplicant income fields include a tooltip clarifying that the value should
be entered as a raw number (for example 5000), not in hundreds or thousands.
Loan Amount Term is a plain numeric field (in months) rather than a dropdown.
If any field is left empty, the app shows a warning instead of predicting.
Every completed prediction is logged automatically.

**About Us**
Describes what the project does, how the underlying model was built, and who
it is intended for, followed by a contact form (name, email, message) for
questions or feedback. Submitted messages are validated and logged.

To run the app, make sure the notebook has been run at least once so that the
`models` folder contains the three pickle files, then from the project root run:

```
streamlit run app.py
```

This will open the app in your browser, usually at `http://localhost:8501`.

## Logging

The `logs` folder is created automatically if it does not exist, and is kept
permanently in the project so history is not lost between runs.

- `logs/app.log` records every app event (model load, predictions made,
  contact messages, errors) with a timestamp.
- `logs/predictions_log.csv` stores a structured record of every prediction
  made, including all input values, the predicted status, and the approval
  probability. This file powers the Dashboard page.
- `logs/contact_messages.csv` stores every message submitted through the
  About Us contact form.

