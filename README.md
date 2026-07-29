# Loan Default Prediction

A simple machine learning project that predicts whether a loan application will
be approved or not, based on applicant details such as income, credit history,
education, and property area.

## Project Overview

This project covers the full workflow of a classification problem:

- Exploratory Data Analysis (EDA) and visualization, including deeper
  insight-driven charts (Credit History vs Loan Status, Income and Loan
  Amount vs Loan Status, Approval Rate by Property Area)
- Data cleaning and missing value handling
- Exploratory feature engineering (Total Income, log transform) and a
  dedicated Key Insights summary
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
|-- charts/
|   |-- *.png                     All EDA and model comparison charts, saved automatically when the notebook runs
|
|-- data/
|   |-- raw/
|   |   |-- loan_dataset.csv          Raw dataset
|   |-- processed/
|   |   |-- processed_loan_dataset.csv Cleaned and encoded dataset
|
|-- logs/
|   |-- app.log                   Text log of every app event
|   |-- predictions_log.csv       Structured log of every prediction made
|   |
|-- models/
|   |-- best_model.pkl            Trained model with the best test accuracy
|   |-- scaler.pkl                StandardScaler fitted on training data
|   |-- encoder.pkl               LabelEncoders for categorical columns and target
|
|-- notebook/
|   |-- loan_default_prediction.ipynb   Main notebook with the full workflow
|
|-- styles/
|   |-- style.css                 All custom CSS for the Streamlit app
|
|-- app.py                        Streamlit web app for loan status prediction
|-- loan_default_prediction.html         HTML export of the notebook
|-- loan_default_prediction_notebook.pdf PDF export of the notebook
|-- loan_default_prediction_summary.pdf  One-page project summary
|-- requirements.txt
|-- README.md
```

Note: the notebook resolves `data/`, `charts/`, and `models/` as siblings of
the `notebook/` folder (one level up from where the notebook itself lives),
so keep this folder layout intact when running it.

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

The raw file has 614 rows and 13 columns, with missing values in `Gender`
(13), `Married` (3), `Dependents` (15), `Self_Employed` (32), `LoanAmount`
(22), `Loan_Amount_Term` (14), and `Credit_History` (50).

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
2. Load the dataset and take a first look (shape, info, describe)
3. Initial EDA on the raw data: target variable distribution and a missing
   values heatmap
4. Check and handle missing values (mode for categorical, median for
   numerical)
5. Drop irrelevant columns (`Loan_ID`)
6. Continue EDA on the now-clean data: categorical features vs Loan Status,
   numerical feature distributions, outlier boxplots, and a correlation
   heatmap
7. Deeper, more insightful EDA: Credit History vs Loan Status, Income and
   Loan Amount vs Loan Status, and Loan Approval Rate by Property Area
8. Exploratory feature engineering: a combined Total Income feature and a
   log transform to reduce its skew (kept separate from the modeling data
   so the existing input pipeline is unaffected)
9. Key Insights from EDA summary
10. Encode categorical features
11. Save the processed dataset
12. Split dependent and independent features
13. Train test split
14. Feature scaling
15. Train models with GridSearchCV (Logistic Regression, Random Forest, SVM,
    KNN, XGBoost)
16. Compare model results, including a bar chart of train vs test accuracy
17. Select the best model
18. Export the best model, scaler, and encoders
19. Test the best model with custom input

Every chart in the notebook is also saved automatically as a PNG file in the
`charts/` folder as it is generated.

## Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

Each model is tuned using `GridSearchCV` with 5-fold cross validation, and
compared on both training and test accuracy to check for overfitting. On the
current dataset, **Logistic Regression** comes out as the best model, with
80.04% training accuracy and 86.18% test accuracy.

## Key Insights from EDA

- **Class balance:** about 68.7% of applications were approved (`Y`) and
  31.3% were not (`N`).
- **Credit history is the dominant signal:** applicants with a credit
  history were approved 79.0% of the time, versus only 7.9% for those
  without one — by far the strongest relationship in the data.
- **Property area matters moderately:** Semiurban applicants had the
  highest approval rate (76.8%), ahead of Urban (65.8%) and Rural (61.5%).
- **Education has a smaller effect:** Graduates were approved 70.8% of the
  time versus 61.2% for non-graduates.
- **Income alone is a weak predictor:** mean Applicant Income was nearly
  identical between approved (5,384) and not-approved (5,446) applicants,
  and the same holds for Total Income (6,889 vs 7,324).
- **Income and loan amount are heavily skewed:** Applicant Income has a
  skew of 6.54 and Total Income 5.63; a log transform brings Total Income's
  skew down to 1.08.
- **Outliers are present** in `ApplicantIncome`, `CoapplicantIncome`, and
  `LoanAmount`, consistent with the skew above.

## Output Files

After running the notebook, the following files are created inside the
`models` folder:

- `best_model.pkl` : the model with the highest test accuracy
- `scaler.pkl` : the StandardScaler fitted on the training data
- `encoder.pkl` : a dictionary of LabelEncoders for each categorical column
  and the target column

These three files are all that is needed to make predictions on new data,
for example inside a Streamlit web app. The `charts/` folder additionally
collects every plot generated during the notebook run as a PNG file, and
`data/processed/processed_loan_dataset.csv` holds the cleaned and encoded
dataset.

## Notebook Exports and Project Summary

Alongside the executable notebook, the project includes ready-to-share
versions of the results:

- `loan_default_prediction.html` : full HTML export of the executed notebook
  (all code, outputs, and charts), viewable in any browser.
- `loan_default_prediction_notebook.pdf` : PDF export of the same notebook,
  for printing or sharing where a browser isn't convenient.
- `loan_default_prediction_summary.pdf` : a one-page summary covering the
  project goal, the chosen model (Logistic Regression) with its final
  training and test accuracy, and the key insights from the EDA (credit
  history as the dominant predictor, class imbalance, and the roles of
  income, property area, and education).

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
