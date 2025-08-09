import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import subprocess
import sys

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore')

@st.cache_resource
def install_dependencies():
    try:
        # Attempt to import a key library to check if dependencies are met
        import sklearn
        import imblearn
        import xgboost
        import plotly
        st.success("Dependencies already met.")
    except ImportError:
        st.warning("Installing missing dependencies. This may take a moment...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.2.2", "imbalanced-learn==0.10.1", "xgboost", "plotly", "pandas", "joblib"])
            st.success("Dependencies installed successfully!")
        except Exception as e:
            st.error(f"Failed to install dependencies: {e}. Please ensure you have pip installed and an active internet connection.")
            st.stop() # Stop the app if dependencies can't be installed

# Run dependency installation only once
install_dependencies()

# --- Model Asset Loading (Requires notebook to be run first) ---
@st.cache_resource
def load_model_assets():
    """Loads the pre-trained model pipeline and LabelEncoder."""
    try:
        pipeline = joblib.load('sme_loan_preapproval_model.pkl')
        le = joblib.load('label_encoder.pkl')
        return pipeline, le
    except FileNotFoundError:
        st.error(
            "Model files ('sme_loan_preapproval_model.pkl', 'label_encoder.pkl') not found. "
            "Please ensure you have run the entire Jupyter Notebook first to generate and save them."
        )
        st.stop() # Stop the app if model files are missing
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.stop()

# --- Synthetic Dataset Generation (for EDA) ---
@st.cache_data
def generate_synthetic_dataset():
    """Generates the synthetic dataset for display."""
    n_samples = 5000

    annual_revenue = np.random.uniform(50000, 5000000, n_samples)
    years_in_business = np.random.uniform(1, 30, n_samples)
    credit_score = np.random.uniform(500, 850, n_samples)
    avg_monthly_turnover = annual_revenue / 12 * np.random.uniform(0.8, 1.2, n_samples)
    existing_loans_count = np.random.randint(0, 5, n_samples)
    industry_risk_rating = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.35, 0.15])
    collateral_value = np.random.uniform(0, 5000000, n_samples)
    requested_amount = np.random.uniform(5000, 1000000, n_samples)
    repayment_term_months = np.random.choice([12, 24, 36, 48, 60], n_samples)
    past_defaults = np.random.randint(0, 2, n_samples)
    relationship_with_bank_years = np.random.uniform(0, 20, n_samples)

    df = pd.DataFrame({
        'annual_revenue': annual_revenue,
        'years_in_business': years_in_business,
        'credit_score': credit_score,
        'avg_monthly_turnover': avg_monthly_turnover,
        'existing_loans_count': existing_loans_count,
        'industry_risk_rating': industry_risk_rating,
        'collateral_value': collateral_value,
        'requested_amount': requested_amount,
        'repayment_term_months': repayment_term_months,
        'past_defaults': past_defaults,
        'relationship_with_bank_years': relationship_with_bank_years
    })

    def get_loan_status(row):
        score = 0
        if row['credit_score'] > 700: score += 2
        if row['years_in_business'] > 5: score += 1
        if row['annual_revenue'] > 500000: score += 1
        if row['collateral_value'] > row['requested_amount'] * 0.8: score += 1
        if row['past_defaults'] == 0: score += 2
        if row['industry_risk_rating'] == 'Low': score += 1
        if row['credit_score'] < 600: score -= 2
        if row['requested_amount'] > row['annual_revenue'] * 0.5: score -= 1
        if row['industry_risk_rating'] == 'High': score -= 2
        if row['past_defaults'] > 0: score -= 3
        score += np.random.uniform(-2, 2)
        return 'Pre-Approved' if score > 3 else 'Needs Manual Review'

    df['loan_status'] = df.apply(get_loan_status, axis=1)
    df_preapproved = df[df['loan_status'] == 'Pre-Approved'].sample(frac=0.8, random_state=42)
    df_manual_review = df[df['loan_status'] == 'Needs Manual Review']
    df = pd.concat([df_preapproved, df_manual_review]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# --- Prediction Function ---
def predict_preapproval(applicant_data_dict: dict, pipeline, le) -> tuple[str, str]:
    """
    Predicts the loan status for a single SME applicant and provides a reason.
    """
    applicant_df = pd.DataFrame([applicant_data_dict])
    prediction_encoded = pipeline.predict(applicant_df)[0]
    prediction = le.inverse_transform([prediction_encoded])[0]

    reason_points = []
    
    if prediction == 'Pre-Approved':
        reason_points.append("The application shows strong potential for pre-approval due to several positive indicators:")
        if applicant_data_dict['credit_score'] >= 700:
            reason_points.append(f"- **Excellent Credit Score ({applicant_data_dict['credit_score']}):** A high credit score indicates strong financial reliability.")
        elif applicant_data_dict['credit_score'] >= 650:
            reason_points.append(f"- **Good Credit Score ({applicant_data_dict['credit_score']}):** A solid credit history supports the application.")
        
        if applicant_data_dict['years_in_business'] >= 5:
            reason_points.append(f"- **Established Business (Years in Business: {int(applicant_data_dict['years_in_business'])}):** A longer operational history signifies stability.")
        
        if applicant_data_dict['annual_revenue'] >= 750000:
            reason_points.append(f"- **Substantial Annual Revenue (R{applicant_data_dict['annual_revenue']:,}):** High revenue demonstrates robust business activity.")
        elif applicant_data_dict['annual_revenue'] >= 300000:
            reason_points.append(f"- **Strong Annual Revenue (R{applicant_data_dict['annual_revenue']:,}):** Consistent revenue flow is a positive sign.")

        if applicant_data_dict['industry_risk_rating'] == 'Low':
            reason_points.append("- **Low Industry Risk:** Operating in a low-risk industry enhances the application's profile.")
        
        if applicant_data_dict['past_defaults'] == 0:
            reason_points.append("- **No History of Defaults:** A clean record of no past loan defaults is a significant positive factor.")
        
        if applicant_data_dict['collateral_value'] >= applicant_data_dict['requested_amount'] * 0.8:
            reason_points.append(f"- **Strong Collateral (R{applicant_data_dict['collateral_value']:,}):** The provided collateral provides good security relative to the requested amount.")

    else: # Needs Manual Review
        reason_points.append("The application requires manual review due to certain factors that raise concerns or necessitate further assessment:")
        if applicant_data_dict['credit_score'] < 600:
            reason_points.append(f"- **Lower Credit Score ({applicant_data_dict['credit_score']}):** A credit score below the desired threshold may indicate higher risk.")
        elif applicant_data_dict['credit_score'] < 650:
             reason_points.append(f"- **Moderate Credit Score ({applicant_data_dict['credit_score']}):** While not critical, a moderate credit score could benefit from deeper analysis.")
        
        if applicant_data_dict['years_in_business'] < 3:
            reason_points.append(f"- **Newer Business (Years in Business: {int(applicant_data_dict['years_in_business'])}):** A shorter business history means less operational data for assessment.")
        
        if applicant_data_dict['industry_risk_rating'] == 'High':
            reason_points.append("- **High Industry Risk:** The industry sector carries an elevated risk profile, requiring closer scrutiny.")
        elif applicant_data_dict['industry_risk_rating'] == 'Medium':
            reason_points.append("- **Medium Industry Risk:** This industry carries some risk, warranting a manual review for a comprehensive understanding.")
        
        if applicant_data_dict['past_defaults'] > 0:
            reason_points.append(f"- **History of Past Defaults ({int(applicant_data_dict['past_defaults'])}):** Previous defaults are a significant red flag that requires detailed investigation.")
        
        if applicant_data_dict['requested_amount'] > applicant_data_dict['annual_revenue'] * 0.7:
            reason_points.append(f"- **High Requested Amount relative to Revenue (R{applicant_data_dict['requested_amount']:,} requested vs. R{applicant_data_dict['annual_revenue']:,} annual revenue):** The loan amount requested is proportionally high compared to the business's annual revenue.")
        
        if applicant_data_dict['existing_loans_count'] >= 3:
            reason_points.append(f"- **Multiple Existing Loans ({int(applicant_data_dict['existing_loans_count'])}):** A high number of existing loans might indicate over-leveraging.")

    return prediction, "\n".join(reason_points)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Instant SME Loan Pre-Approval System", layout="centered")

st.title("Instant SME Loan Pre-Approval System")
st.markdown("""
Welcome to the Instant SME Loan Pre-Approval System! This proof-of-concept application
demonstrates how a machine learning model can rapidly assess SME loan applications,
classifying them as 'Pre-Approved' or 'Needs Manual Review'.
""")

# --- EDA Section ---
st.header("Exploratory Data Analysis (EDA)")
st.markdown("Here is a look at the synthetic dataset used to train the model. The interactive plots help us understand the distribution of key features and the class balance.")

df = generate_synthetic_dataset()

st.subheader("Dataset Preview and Statistics")
st.write(df.head())
st.write(df.describe().T)

st.subheader("Data Distributions")
fig = make_subplots(rows=2, cols=3, subplot_titles=(
    'Distribution of Annual Revenue', 'Distribution of Credit Score',
    'Industry Risk Rating Counts', 'Distribution of Years in Business',
    'Distribution of Requested Amount', 'Loan Status Class Balance'
))
fig.add_trace(go.Histogram(x=df['annual_revenue'], name='Annual Revenue'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['credit_score'], name='Credit Score'), row=1, col=2)
fig.add_trace(go.Bar(x=df['industry_risk_rating'].value_counts().index,
                     y=df['industry_risk_rating'].value_counts().values, name='Industry Risk'), row=1, col=3)
fig.add_trace(go.Histogram(x=df['years_in_business'], name='Years in Business'), row=2, col=1)
fig.add_trace(go.Histogram(x=df['requested_amount'], name='Requested Amount'), row=2, col=2)
fig.add_trace(go.Bar(x=df['loan_status'].value_counts().index,
                     y=df['loan_status'].value_counts().values, name='Loan Status'), row=2, col=3)
fig.update_layout(height=800, width=1200, title_text="Data Exploration with Plotly", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Class Balance")
st.write("The distribution of the target variable shows the number of 'Pre-Approved' vs. 'Needs Manual Review' applications.")
st.dataframe(df['loan_status'].value_counts().to_frame())

st.markdown("---")

# --- Applicant Input Section ---
st.header("Applicant Information")
st.markdown("Fill out the details below to receive an instant pre-approval decision.")

col1, col2 = st.columns(2)

with col1:
    annual_revenue = st.number_input("Annual Revenue (R)", min_value=10000.0, max_value=10000000.0, value=500000.0, step=10000.0)
    years_in_business = st.number_input("Years in Business", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    credit_score = st.number_input("Credit Score (500-850)", min_value=500, max_value=850, value=700, step=1)
    avg_monthly_turnover = st.number_input("Average Monthly Turnover (R)", min_value=1000.0, max_value=1000000.0, value=40000.0, step=1000.0)
    existing_loans_count = st.number_input("Existing Loans Count", min_value=0, max_value=10, value=0, step=1)

with col2:
    industry_risk_rating = st.selectbox("Industry Risk Rating", ['Low', 'Medium', 'High'])
    collateral_value = st.number_input("Collateral Value (R)", min_value=0.0, max_value=10000000.0, value=200000.0, step=10000.0)
    requested_amount = st.number_input("Requested Amount (R)", min_value=1000.0, max_value=5000000.0, value=100000.0, step=1000.0)
    repayment_term_months = st.selectbox("Repayment Term (Months)", [12, 24, 36, 48, 60])
    past_defaults = st.selectbox("Past Defaults", [0, 1, 2, 3])
    relationship_with_bank_years = st.number_input("Relationship with Bank (Years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)

# --- Prediction & Explanation ---
st.markdown("---")
st.header("Prediction")

# Load model assets just before prediction to ensure they're available
pipeline, le = load_model_assets()

if st.button("Get Pre-Approval Status"):
    applicant_data = {
        'annual_revenue': annual_revenue,
        'years_in_business': years_in_business,
        'credit_score': credit_score,
        'avg_monthly_turnover': avg_monthly_turnover,
        'existing_loans_count': existing_loans_count,
        'industry_risk_rating': industry_risk_rating,
        'collateral_value': collateral_value,
        'requested_amount': requested_amount,
        'repayment_term_months': repayment_term_months,
        'past_defaults': past_defaults,
        'relationship_with_bank_years': relationship_with_bank_years
    }

    status, reason = predict_preapproval(applicant_data, pipeline, le)

    st.subheader("Pre-Approval Decision:")
    if status == "Pre-Approved":
        st.success(f"**{status}**")
    else:
        st.warning(f"**{status}**")
    
    st.subheader("Reasoning:")
    st.markdown(reason)

st.markdown("---")
st.markdown("""
This is a proof-of-concept. For a real-world banking system, further
validation, regulatory compliance, and integration with actual customer data are essential.
""")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #e6e6e6;
    }
    </style>
    <div class="footer">
        Developed By Andiswa Mabuza | Email: <a href="mailto:Amabuza53@gmail.com">Amabuza53@gmail.com</a> | Developer Site: <a href="https://andiswamabuza.vercel.app">https://andiswamabuza.vercel.app</a>
    </div>
    """,
    unsafe_allow_html=True
)
