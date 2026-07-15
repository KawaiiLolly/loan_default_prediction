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
|-- data/
|   |-- loan_dataset.csv          Raw dataset
|   |-- processed_loan_data.csv   Cleaned and encoded dataset
|
|-- models/
|   |-- best_model.pkl            Trained model with the best test accuracy
|   |-- scaler.pkl                StandardScaler fitted on training data
|   |-- encoders.pkl              LabelEncoders for categorical columns and target
|
|-- notebook/
|   |-- loan_default_prediction.ipynb   Main notebook with the full workflow
|
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

## Next Steps

- Build a Streamlit web app that loads `best_model.pkl`, `scaler.pkl`, and
  `encoders.pkl` to take user input from a form and display the prediction.