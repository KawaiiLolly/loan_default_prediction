# Libraries, Dependencies
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime
import plotly.express as px

# Page config and theme
st.set_page_config(page_title="Loan Default Prediction", page_icon=None, layout="wide")

# Folder setup
LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)

APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
PREDICTIONS_LOG_FILE = os.path.join(LOG_DIR, "predictions_log.csv")
CONTACT_LOG_FILE = os.path.join(LOG_DIR, "contact_messages.csv")

PREDICTION_COLUMNS = [
    "Timestamp", "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "Prediction", "Approval_Probability"
]

CONTACT_COLUMNS = ["Timestamp", "Name", "Email", "Message"]
CHOOSE_OPTION = "Choose an option"

# Logging setup
logging.basicConfig(
    filename=APP_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger("loan_app")

def init_csv_log(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        logger.info(f"Created a new log file: {path}")

def log_prediction(record):
    row = pd.DataFrame([record], columns=PREDICTION_COLUMNS)
    row.to_csv(PREDICTIONS_LOG_FILE, mode="a", header=False, index=False)
    logger.info(f"Prediction logged: {record}")

def log_contact_message(record):
    row = pd.DataFrame([record], columns=CONTACT_COLUMNS)
    row.to_csv(CONTACT_LOG_FILE, mode="a", header=False, index=False)
    logger.info(f"Contact message logged from: {record.get('Email')}")

def load_predictions_log():
    if os.path.exists(PREDICTIONS_LOG_FILE):
        try:
            return pd.read_csv(PREDICTIONS_LOG_FILE)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


init_csv_log(PREDICTIONS_LOG_FILE, PREDICTION_COLUMNS)
init_csv_log(CONTACT_LOG_FILE, CONTACT_COLUMNS)

# Load model artifacts
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


try:
    model, scaler, encoders = load_artifacts()
    logger.info("Model artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    st.error("Could not load the model files. Please make sure the notebook has been run at least once.")
    st.stop()

cat_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

column_order = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

GREEN_DARK = "#1B3B2F"
GREEN_MID = "#2F6B4F"
RED_SOFT = "#C0524B"

def load_css(path):
    with open(path) as f:
        return f.read()

CSS_FILE = os.path.join("styles", "style.css")
st.markdown(f"<style>{load_css(CSS_FILE)}</style>", unsafe_allow_html=True)

PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}

def style_chart(fig, height=320):
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Poppins, sans-serif", color="#33443A", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

# Helper functions
def predict_loan_status(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[column_order]

    for col in cat_features:
        input_df[col] = encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prediction_label = encoders["Loan_Status"].inverse_transform([prediction])[0]

    approval_probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        approved_index = list(model.classes_).index(1)
        approval_probability = round(proba[approved_index] * 100, 2)

    return prediction_label, approval_probability


def stat_card(col, value, label):
    col.markdown(
        f"""<div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>""",
        unsafe_allow_html=True
    )

# Navigation state
NAV_ITEMS = [
    ("Dashboard", ":material/space_dashboard:"),
    ("Prediction", ":material/insights:"),
    ("About Us", ":material/info:"),
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


def render_bottom_nav(container_key):
    nav_container = st.container(key=container_key)
    with nav_container:
        nav_cols = st.columns(len(NAV_ITEMS))
        for col, (page_name, icon) in zip(nav_cols, NAV_ITEMS):
            is_active = st.session_state.page == page_name
            btn_type = "primary" if is_active else "secondary"
            if col.button(page_name, key=f"{container_key}_{page_name}", icon=icon, type=btn_type):
                if st.session_state.page != page_name:
                    st.session_state.page = page_name
                    st.rerun()


render_bottom_nav("bottom_nav_full")
render_bottom_nav("bottom_nav_compact")

page = st.session_state.page

# Dashboard
if page == "Dashboard":

    log_df = load_predictions_log()
    total_predictions = len(log_df)
    approved_count = int((log_df["Prediction"] == "Y").sum()) if total_predictions else 0
    approval_rate = round((approved_count / total_predictions) * 100, 1) if total_predictions else 0.0

    st.markdown(
        f"""<div class="hero-card">
                <div class="eyebrow">Welcome</div>
                <div class="headline">Loan Default Prediction</div>
                <div class="subtext">{total_predictions} predictions made so far, {approval_rate}% approval rate</div>
            </div>""",
        unsafe_allow_html=True
    )

    if log_df.empty:
        st.info("No predictions yet. Go to the 'Prediction' page to make your first prediction.")
    else:
        total = len(log_df)
        approved = int((log_df["Prediction"] == "Y").sum())
        rejected = total - approved
        rate = round((approved / total) * 100, 1)

        st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        stat_card(c1, total, "Total Predictions")
        stat_card(c2, approved, "Approved")
        stat_card(c3, rejected, "Not Approved")
        stat_card(c4, f"{rate}%", "Approval Rate")

        st.markdown('<div class="section-title">Recent Activity (last 10 predictions)</div>', unsafe_allow_html=True)

        recent = log_df.tail(10).iloc[::-1]

        for _, row in recent.iterrows():
            badge_class = "badge-approved" if row["Prediction"] == "Y" else "badge-rejected"
            badge_text = "Approved" if row["Prediction"] == "Y" else "Not Approved"

            st.markdown(
                f"""<div class="activity-card">
                        <div class="activity-left">
                            <div class="activity-title">
                                {row['Gender']}, {row['Married']} | Income: {row['ApplicantIncome']}
                                | Loan: {row['LoanAmount']}k | {row['Property_Area']}
                            </div>
                            <div class="activity-sub">{row['Timestamp']}</div>
                        </div>
                        <div class="activity-right">
                            <div class="activity-amount">{row['Approval_Probability']}%</div>
                            <span class="badge {badge_class}">{badge_text}</span>
                        </div>
                    </div>""",
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">EDA on Prediction History</div>', unsafe_allow_html=True)
        st.caption("Charts are interactive. Hover for details, drag to zoom, and click legend items to toggle.")

        color_map = {"Approved": GREEN_MID, "Not Approved": RED_SOFT}

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="chart-card"><div class="chart-title">Approved vs Not Approved</div>', unsafe_allow_html=True)
            counts = log_df["Prediction"].value_counts()
            labels = ["Approved" if i == "Y" else "Not Approved" for i in counts.index]

            fig1 = px.pie(
                values=counts.values,
                color=labels,
                color_discrete_map=color_map,
                hole=0.45
            )
            fig1 = style_chart(fig1)
            st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="chart-card"><div class="chart-title">Credit History vs Prediction</div>', unsafe_allow_html=True)
            ch_data = log_df.copy()
            ch_data["Credit_History"] = ch_data["Credit_History"].map(
                {1.0: "Has Credit History", 0.0: "No Credit History"}
            )
            ch_data["Prediction_Label"] = ch_data["Prediction"].map({"Y": "Approved", "N": "Not Approved"})

            fig2 = px.histogram(
                ch_data,
                x="Credit_History",
                color="Prediction_Label",
                barmode="group",
                color_discrete_map=color_map
            )
            fig2.update_layout(xaxis_title="", yaxis_title="Count", legend_title="")
            fig2 = style_chart(fig2)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown('</div>', unsafe_allow_html=True)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="chart-card"><div class="chart-title">Applicant Income Distribution</div>', unsafe_allow_html=True)
            fig3 = px.histogram(
                log_df,
                x="ApplicantIncome",
                nbins=10,
                color_discrete_sequence=[GREEN_MID]
            )
            fig3.update_layout(xaxis_title="Applicant Income", yaxis_title="Count")
            fig3 = style_chart(fig3)
            st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="chart-card"><div class="chart-title">Approval Rate by Property Area</div>', unsafe_allow_html=True)
            area_data = log_df.copy()
            area_data["Approved"] = (area_data["Prediction"] == "Y").astype(int)
            area_rate = area_data.groupby("Property_Area")["Approved"].mean().reset_index()
            area_rate["Approved"] = area_rate["Approved"] * 100

            fig4 = px.bar(
                area_rate,
                x="Property_Area",
                y="Approved",
                color_discrete_sequence=[GREEN_DARK]
            )
            fig4.update_layout(xaxis_title="", yaxis_title="Approval Rate (%)", yaxis_range=[0, 100])
            fig4 = style_chart(fig4)
            st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown('</div>', unsafe_allow_html=True)

# Prediction
elif page == "Prediction":

    st.markdown('<div class="page-title">Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Enter applicant details to check the loan approval outcome.</div>',
        unsafe_allow_html=True
    )

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", [CHOOSE_OPTION] + list(encoders["Gender"].classes_))
            married = st.selectbox("Married", [CHOOSE_OPTION] + list(encoders["Married"].classes_))
            dependents = st.selectbox("Dependents", [CHOOSE_OPTION] + list(encoders["Dependents"].classes_))
            education = st.selectbox("Education", [CHOOSE_OPTION] + list(encoders["Education"].classes_))
            self_employed = st.selectbox("Self Employed", [CHOOSE_OPTION] + list(encoders["Self_Employed"].classes_))
            property_area = st.selectbox("Property Area", [CHOOSE_OPTION] + list(encoders["Property_Area"].classes_))

        with col2:
            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0,
                value=None,
                step=500,
                placeholder="e.g. 5000",
                help="Enter the applicant's monthly income as a raw number, for example 5000 for five thousand. Do not enter it in hundreds or thousands."
            )
            coapplicant_income = st.number_input(
                "Coapplicant Income",
                min_value=0,
                value=None,
                step=500,
                placeholder="e.g. 1500",
                help="Enter the co-applicant's monthly income as a raw number, for example 1500. Enter 0 if there is no co-applicant."
            )
            loan_amount = st.number_input(
                "Loan Amount (in thousands)",
                min_value=0,
                value=None,
                step=1,
                placeholder="e.g. 128",
                help="Enter the loan amount in thousands, for example 128 for a loan of 128,000."
            )
            loan_amount_term = st.number_input(
                "Loan Amount Term (in months)",
                min_value=0,
                value=None,
                step=12,
                placeholder="e.g. 360",
                help="Common terms are 12, 36, 60, 84, 120, 180, 240, 300, or 360 months."
            )
            credit_history = st.selectbox("Credit History", [CHOOSE_OPTION, "Yes", "No"])

        submitted = st.form_submit_button("Predict Loan Status")

    if submitted:
        raw_values = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }

        missing_fields = [
            k for k, v in raw_values.items() if v is None or v == CHOOSE_OPTION
        ]

        if missing_fields:
            st.warning("Please fill in all fields before predicting.")
        else:
            credit_history_value = 1.0 if credit_history == "Yes" else 0.0

            input_data = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history_value,
                "Property_Area": property_area
            }

            try:
                prediction_label, approval_probability = predict_loan_status(input_data)

                record = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **input_data,
                    "Prediction": prediction_label,
                    "Approval_Probability": approval_probability
                }
                log_prediction(record)

                st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

                if prediction_label == "Y":
                    st.markdown(
                        f'<div class="result-approved">The loan is likely to be Approved. '
                        f'Approval Probability: {approval_probability} %</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-rejected">The loan is likely to be Not Approved. '
                        f'Approval Probability: {approval_probability} %</div>',
                        unsafe_allow_html=True
                    )

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                st.error("Something went wrong while making the prediction. Please check the input values.")

# About Us
elif page == "About Us":

    st.markdown('<div class="page-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">What Loan Default Prediction does and how it works.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """<div class="about-card">
                <h4>What this project does</h4>
                <p>
                Loan Default Prediction is a machine learning tool that estimates whether a
                loan application is likely to be approved. It looks at applicant details such
                as income, credit history, education, employment status, and property area,
                and produces an instant approval prediction along with a confidence score.
                </p>
                <h4>How it works</h4>
                <p>
                The underlying model was trained on historical loan application data. The data
                was cleaned, missing values were handled, categorical fields were encoded, and
                several algorithms (Logistic Regression, Random Forest, SVM, KNN, and XGBoost)
                were compared using cross validated hyperparameter tuning. The best performing
                model, along with its encoders and scaler, powers the Prediction page.
                </p>
                <h4>Who it is for</h4>
                <p>
                This tool is meant as a quick, internal screening aid to get a fast read on an
                application. It is not a substitute for full underwriting or a final lending
                decision, and results should always be reviewed by a qualified professional.
                </p>
            </div>""",
        unsafe_allow_html=True
    )
