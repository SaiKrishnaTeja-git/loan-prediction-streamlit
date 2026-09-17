import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier


# ============================================================
# OPTIONAL GROQ IMPORT
# ============================================================

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 16px;
        background: linear-gradient(135deg,#111827,#243b53);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        margin-bottom: 6px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
    }

    .result-box {
        padding: 22px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        margin-top: 18px;
    }

    .ai-box {
        padding: 20px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #dbeafe;
        margin: 12px 0;
    }

    .small-note {
        font-size: 0.9rem;
        color: #64748b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "07_loan_train.xls"


# ============================================================
# MODEL FEATURES
# ============================================================

# Customer-facing reduced feature set.
#
# IMPORTANT:
# This is a portfolio/demo model.
# It should be retrained and validated on this exact
# feature set before any real-world deployment.

NUMERIC = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

CATEGORICAL = [
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

FEATURES = NUMERIC + CATEGORICAL


# ============================================================
# GROQ CONFIGURATION
# ============================================================

# IMPORTANT:
# We deliberately hard-code the current model here so an old
# GROQ_MODEL environment variable cannot override it.
#
# Do NOT put your API key in this file.

GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_client():
    """
    Create a Groq client without exposing the API key
    in source code.
    """

    if Groq is None:
        return None

    # --------------------------------------------------------
    # Streamlit Cloud / deployed application
    # --------------------------------------------------------

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    # --------------------------------------------------------
    # Local development
    # --------------------------------------------------------

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_demo_model():

    # --------------------------------------------------------
    # Read dataset
    # --------------------------------------------------------

    df = pd.read_excel(DATA_FILE)

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    X = df[FEATURES].copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = (df["Loan_Status"] == "Y").astype(int)

    # --------------------------------------------------------
    # Clean Dependents
    # --------------------------------------------------------

    X["Dependents"] = (
        X["Dependents"]
        .astype(str)
        .replace({
            "nan": np.nan,
            "None": np.nan
        })
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC
            ),

            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent")
                        ),

                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )
                    ]
                ),
                CATEGORICAL
            )
        ]
    )

    # --------------------------------------------------------
    # XGBoost Model
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    # --------------------------------------------------------
    # Complete Pipeline
    # --------------------------------------------------------

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipe.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pred = pipe.predict(X_test)

    prob = pipe.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, pred)

    auc = roc_auc_score(y_test, prob)

    return pipe, accuracy, auc


# ============================================================
# AI PROMPT
# ============================================================

def build_ai_prompt(application, probability, prediction):
    """
    Build a grounded prompt.

    The LLM explains the model output.
    It does NOT make the credit decision.
    """

    outcome = (
        "indicative approved"
        if prediction
        else "indicative not approved"
    )

    return f"""
You are an AI financial guidance assistant inside a portfolio
demonstration called Loan Approval Predictor.

Your job is to turn the supplied application data and ML output
into useful, personalized, plain-English guidance.

IMPORTANT RULES:

- Do NOT make or claim to make a real lending decision.
- Do NOT override the ML result.
- Do NOT invent credit score information.
- Do NOT invent credit-bureau information.
- Do NOT invent existing debts.
- Do NOT invent interest rates.
- Do NOT invent assets.
- Do NOT invent employer details.
- Do NOT invent salary details beyond what is supplied.
- Do NOT invent any other missing financial facts.
- Do not use gender or marital status.
- Clearly label assumptions or illustrative examples.

The model output is {outcome} with a model probability of
{probability:.1%}.

This is an illustrative model probability,
NOT a guaranteed approval probability.

APPLICATION DATA:

- Monthly applicant income: ₹{application['monthly_income']:,.0f}
- Monthly co-applicant income: ₹{application['monthly_coapp_income']:,.0f}
- Combined monthly income: ₹{application['total_monthly']:,.0f}
- Requested loan amount: ₹{application['requested_loan']:,.0f}
- Preferred repayment term: {application['term']} months
- Dependents: {application['dependents']}
- Education: {application['education']}
- Employment: {application['employment']}
- Property area: {application['property_area']}

Create a personalized plan using EXACTLY these sections:

1. What the model is saying

Explain the model result carefully without pretending
the model knows facts it does not know.

2. Application strengths

Identify positive aspects visible from the supplied inputs.

3. Potential pressure points

Identify affordability/application factors that may need
attention.

Avoid unsupported claims.

4. Personalized action plan

Give 3 to 5 practical, non-guaranteed actions.

Examples:

- Review the requested loan amount.
- Compare repayment terms.
- Validate income documentation.
- Consider whether a co-applicant is appropriate.
- Check credit-bureau information before applying.

5. Loan planning scenarios

Give 2 or 3 scenario ideas such as:

- Lower loan amount.
- Different repayment term.
- Stronger documented income.

Do NOT invent interest rates.

Do NOT calculate EMI unless an interest rate is supplied.

6. Before applying

Give a concise checklist of documents/data the applicant
should have ready.

7. Important limitation

Clearly state that final lending decisions can depend on:

- Verified income
- Credit-bureau information
- Affordability
- Lender policy
- Fraud checks
- Other lender-specific criteria

Keep the answer concise, useful and customer-friendly.

Use bullets and ₹ formatting.
"""


# ============================================================
# GENERATE AI INSIGHTS
# ============================================================

def generate_ai_insights(
    application,
    probability,
    prediction
):

    client = get_groq_client()

    if client is None:
        return (
            None,
            "Groq is not configured. Add GROQ_API_KEY "
            "to your environment or Streamlit secrets."
        )

    prompt = build_ai_prompt(
        application,
        probability,
        prediction
    )

    try:

        completion = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You provide grounded financial guidance "
                        "for a loan application demo. "
                        "Be concise, transparent, and never "
                        "invent missing financial facts."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.25,

            max_tokens=1400
        )

        return (
            completion.choices[0].message.content,
            None
        )

    except Exception as e:

        return (
            None,
            f"Groq request failed: {e}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, accuracy, auc = train_demo_model()

    model_ready = True

except Exception as e:

    model_ready = False

    st.error(
        "Could not load 07_loan_train.xls. "
        "Keep it in the same folder as app.py."
    )

    st.exception(e)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🏦 Loan Approval Predictor</h1>
        <p>
            Credit Assessment •
            Machine Learning Insights •
            AI-Powered Applicant Guidance
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.info(
    "Portfolio demonstration only. "
    "The ML model and AI assistant are not a real lending "
    "decision engine."
)


# ============================================================
# TABS
# ============================================================

apply_tab, assessment_tab, ai_tab, insights_tab = st.tabs(
    [
        "📝 Apply",
        "📊 Assessment",
        "🤖 AI Insights",
        "🔍 Model Insights"
    ]
)


# ============================================================
# APPLY TAB
# ============================================================

with apply_tab:

    st.markdown("### Tell us about your loan")

    st.caption(
        "The first screen is intentionally short. "
        "Information that can be retrieved from authorised "
        "internal systems or a credit bureau should not be "
        "repeatedly requested."
    )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    st.markdown("#### Step 1 · Loan & income")

    c1, c2 = st.columns(2)

    with c1:

        monthly_income = st.number_input(
            "Monthly applicant income (₹)",
            min_value=0,
            value=50000,
            step=1000
        )

        monthly_coapp_income = st.number_input(
            "Monthly co-applicant income (₹)",
            min_value=0,
            value=0,
            step=1000
        )

    with c2:

        requested_loan = st.number_input(
            "Requested loan amount (₹)",
            min_value=0,
            value=500000,
            step=10000
        )

        term = st.selectbox(
            "Preferred repayment term (months)",
            [60, 84, 120, 180, 240, 300, 360, 480],
            index=6
        )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    st.markdown("#### Step 2 · Profile")

    c1, c2 = st.columns(2)

    with c1:

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

    with c2:

        employment = st.selectbox(
            "Employment",
            ["Salaried", "Self-employed"]
        )

        property_area = st.selectbox(
            "Property area",
            ["Urban", "Semiurban", "Rural"]
        )

    st.markdown("")

    # --------------------------------------------------------
    # Assessment Button
    # --------------------------------------------------------

    if st.button(
        "🔎 Assess Application",
        type="primary",
        width="stretch"
    ):

        if not model_ready:

            st.error(
                "The model is not available. "
                "Please fix the dataset/model loading issue first."
            )

        else:

            # ------------------------------------------------
            # Convert customer inputs into dataset format
            # ------------------------------------------------

            annual_income = monthly_income * 12

            annual_coapp_income = monthly_coapp_income * 12

            loan_amount_dataset = requested_loan / 1000

            # ------------------------------------------------
            # Create model input
            # ------------------------------------------------

            row = pd.DataFrame(
                [
                    {
                        "ApplicantIncome": annual_income,
                        "CoapplicantIncome": annual_coapp_income,
                        "LoanAmount": loan_amount_dataset,
                        "Loan_Amount_Term": term,
                        "Dependents": dependents,
                        "Education": education,
                        "Self_Employed": (
                            "Yes"
                            if employment == "Self-employed"
                            else "No"
                        ),
                        "Property_Area": property_area
                    }
                ]
            )

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            probability = float(
                model.predict_proba(row)[0, 1]
            )

            prediction = int(
                probability >= 0.50
            )

            # ------------------------------------------------
            # Application object for AI
            # ------------------------------------------------

            application = {

                "monthly_income": monthly_income,

                "monthly_coapp_income": monthly_coapp_income,

                "total_monthly": (
                    monthly_income +
                    monthly_coapp_income
                ),

                "requested_loan": requested_loan,

                "term": term,

                "dependents": dependents,

                "education": education,

                "employment": employment,

                "property_area": property_area
            }

            # ------------------------------------------------
            # Save to session
            # ------------------------------------------------

            st.session_state["row"] = row

            st.session_state["probability"] = probability

            st.session_state["prediction"] = prediction

            st.session_state["application"] = application

            # Clear previous AI response

            st.session_state.pop(
                "ai_insights",
                None
            )

            st.session_state.pop(
                "ai_error",
                None
            )

            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            if prediction == 1:

                st.success(
                    "Indicative model result: APPROVED"
                )

            else:

                st.error(
                    "Indicative model result: NOT APPROVED"
                )

            st.metric(
                "Model probability",
                f"{probability:.1%}"
            )

            st.caption(
                "This probability is generated by the "
                "demonstration ML model and is not a "
                "guaranteed lending outcome."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# ============================================================
# ASSESSMENT TAB
# ============================================================

with assessment_tab:

    st.markdown("### 📊 Application Assessment")

    if "prediction" not in st.session_state:

        st.info(
            "Complete the application first to see "
            "the assessment."
        )

    else:

        probability = st.session_state["probability"]

        prediction = st.session_state["prediction"]

        application = st.session_state["application"]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Model Probability",
            f"{probability:.1%}"
        )

        c2.metric(
            "Requested Loan",
            f"₹{application['requested_loan']:,.0f}"
        )

        c3.metric(
            "Combined Monthly Income",
            f"₹{application['total_monthly']:,.0f}"
        )

        st.markdown("")

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if prediction == 1:

            st.success(
                "The model gives an indicative APPROVED result."
            )

        else:

            st.error(
                "The model gives an indicative NOT APPROVED result."
            )

        st.warning(
            "This is a model output, not a final credit decision. "
            "A real lending workflow would combine verified "
            "information, credit-bureau data, affordability/"
            "policy rules, fraud controls and appropriate review."
        )

        # ----------------------------------------------------
        # Application Summary
        # ----------------------------------------------------

        st.markdown(
            "#### Application summary"
        )

        row = st.session_state["row"].copy()

        # Convert backend values back to customer-friendly units

        row["ApplicantIncome"] *= 1

        row["CoapplicantIncome"] *= 1

        row["LoanAmount"] *= 1000

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # Convert all values to strings to prevent PyArrow
        # mixed-type DataFrame serialization errors.
        # ----------------------------------------------------

        display_row = row.T.astype(str)

        st.dataframe(
            display_row,
            width="stretch"
        )


# ============================================================
# AI INSIGHTS TAB
# ============================================================

with ai_tab:

    st.markdown(
        "### 🤖 Personalized AI Loan Insights"
    )

    st.caption(
        "Groq generates guidance from the application inputs "
        "and the ML output. It does not access your credit "
        "bureau or bank records."
    )

    # --------------------------------------------------------
    # No application yet
    # --------------------------------------------------------

    if "prediction" not in st.session_state:

        st.info(
            "Complete the application first to generate "
            "personalized insights."
        )

    else:

        application = st.session_state["application"]

        probability = st.session_state["probability"]

        prediction = st.session_state["prediction"]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Combined monthly income",
            f"₹{application['total_monthly']:,.0f}"
        )

        c2.metric(
            "Requested loan",
            f"₹{application['requested_loan']:,.0f}"
        )

        c3.metric(
            "Model probability",
            f"{probability:.1%}"
        )

        st.caption(
            f"Powered by Groq • Model: {GROQ_MODEL}"
        )

        # ----------------------------------------------------
        # Groq connection check
        # ----------------------------------------------------

        if get_groq_client() is None:

            st.warning(
                "AI is not connected yet. "
                "Configure your GROQ_API_KEY to enable "
                "personalized insights."
            )

            st.code(
                'Windows PowerShell:\n'
                '$env:GROQ_API_KEY="gsk_your_new_key_here"\n\n'
                'Then restart Streamlit.\n\n'
                'For deployment, use Streamlit Secrets '
                'instead of hardcoding the key.',
                language="text"
            )

        else:

            # ------------------------------------------------
            # Generate AI plan
            # ------------------------------------------------

            if st.button(
                "✨ Generate Personalized Plan",
                type="primary",
                width="stretch"
            ):

                with st.spinner(
                    "Generating personalized insights with Groq..."
                ):

                    answer, error = generate_ai_insights(
                        application,
                        probability,
                        prediction
                    )

                if error:

                    st.session_state["ai_error"] = error

                    st.session_state.pop(
                        "ai_insights",
                        None
                    )

                else:

                    st.session_state["ai_insights"] = answer

                    st.session_state.pop(
                        "ai_error",
                        None
                    )

            # ------------------------------------------------
            # AI Error
            # ------------------------------------------------

            if st.session_state.get("ai_error"):

                st.error(
                    st.session_state["ai_error"]
                )

            # ------------------------------------------------
            # AI Response
            # ------------------------------------------------

            if st.session_state.get("ai_insights"):

                st.markdown(
                    '<div class="ai-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    st.session_state["ai_insights"]
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # Follow-up questions
                # ------------------------------------------------

                st.markdown(
                    "### 💬 Ask a follow-up"
                )

                question = st.text_input(
                    "Ask about this application",
                    placeholder=(
                        "e.g., What could I review before applying?"
                    )
                )

                if st.button(
                    "Ask AI",
                    disabled=not question.strip()
                ):

                    client = get_groq_client()

                    context = build_ai_prompt(
                        application,
                        probability,
                        prediction
                    )

                    follow_up_prompt = f"""
Here is the application context:

{context}

The applicant asks:

{question}

Answer the question directly and concisely.

Rules:

- Use only information available in the context.
- Do not invent credit score information.
- Do not invent debt information.
- Do not invent interest rates.
- Do not make a real lending decision.
- Do not override the ML model.
- Clearly state uncertainty where appropriate.
"""

                    try:

                        completion = client.chat.completions.create(

                            model=GROQ_MODEL,

                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a grounded financial "
                                        "guidance assistant. Answer "
                                        "only from the supplied "
                                        "application context. "
                                        "Do not invent credit data "
                                        "or make final lending "
                                        "decisions."
                                    )
                                },

                                {
                                    "role": "user",
                                    "content": follow_up_prompt
                                }
                            ],

                            temperature=0.25,

                            max_tokens=900
                        )

                        answer = (
                            completion
                            .choices[0]
                            .message
                            .content
                        )

                        st.markdown(
                            '<div class="ai-box">',
                            unsafe_allow_html=True
                        )

                        st.markdown(answer)

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )

                    except Exception as e:

                        st.error(
                            f"Groq request failed: {e}"
                        )


# ============================================================
# MODEL INSIGHTS TAB
# ============================================================

with insights_tab:

    st.markdown(
        "### 🔍 Model Insights"
    )

    if not model_ready:

        st.error(
            "Model is not available."
        )

    else:

        st.markdown(
            "#### Model performance"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Holdout Accuracy",
            f"{accuracy:.1%}"
        )

        c2.metric(
            "ROC-AUC",
            f"{auc:.3f}"
        )

        c3.metric(
            "Model",
            "XGBoost"
        )

        st.markdown("")

        # ----------------------------------------------------
        # Model configuration
        # ----------------------------------------------------

        st.markdown(
            "#### Model configuration"
        )

        model_table = pd.DataFrame(
            {
                "Parameter": [
                    "Algorithm",
                    "Estimators",
                    "Learning Rate",
                    "Max Depth",
                    "Subsample",
                    "Column Sampling",
                    "Decision Threshold"
                ],

                "Value": [
                    "XGBoost",
                    "200",
                    "0.03",
                    "3",
                    "0.80",
                    "0.80",
                    "50%"
                ]
            }
        )

        # Convert everything to string for robust Arrow display

        model_table = model_table.astype(str)

        st.dataframe(
            model_table,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # Feature list
        # ----------------------------------------------------

        st.markdown(
            "#### Features used by the model"
        )

        feature_table = pd.DataFrame(
            {
                "Feature": FEATURES,

                "Type": [
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Categorical",
                    "Categorical",
                    "Categorical",
                    "Categorical"
                ]
            }
        )

        feature_table = feature_table.astype(str)

        st.dataframe(
            feature_table,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # Important limitation
        # ----------------------------------------------------

        st.warning(
            "This portfolio model uses a reduced customer-facing "
            "feature set. The performance metrics shown here are "
            "for demonstration purposes and should not be "
            "interpreted as production credit-risk performance."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Loan Approval Predictor • Portfolio Demonstration • "
    "XGBoost + Groq AI"
)