#%%

import os
import streamlit as st
import shap
import numpy as np
import pandas as pd
import traceback

#%%
# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = os.getcwd()

MODEL_PATH = os.path.join(
    BASE_PATH,
    "results",
    "model",
    "model_28_11_2026_01_18_52",
    "model.pkl"
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "base_data",
    "home_loan_applications_features_20k_extended.csv"
)

# Scorecard scaling
BASE_SCORE = 850
BASE_ODDS = 50
PDO = 50

FACTOR = PDO / np.log(2)

# Class 1 represents cancellation
EVENT_CLASS = 1


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():

    return pd.read_pickle(MODEL_PATH)


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# =========================================================
# SHAP SCORECARD CALCULATION
# =========================================================

def calculate_scorecard(model, input_df):

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(
        model,
        model_output="raw"
    )

    # Calculate SHAP values
    shap_exp = explainer(input_df)

    feature_shap = np.asarray(shap_exp.values)[0]

    base_value = float(
        np.asarray(shap_exp.base_values).reshape(-1)[0]
    )

    # -----------------------------------------------------
    # 1. MODEL PROBABILITY
    # -----------------------------------------------------

    probabilities = model.predict_proba(input_df)[0]

    class_index = list(model.classes_).index(EVENT_CLASS)

    event_probability = float(probabilities[class_index])

    # Avoid division by zero
    event_probability = np.clip(
        event_probability,
        1e-15,
        1 - 1e-15
    )

    # Good : Bad odds
    model_odds = (
        (1 - event_probability) / event_probability
    )

    # -----------------------------------------------------
    # 2. BASELINE ODDS
    # -----------------------------------------------------

    base_odds = np.exp(-base_value)

    # -----------------------------------------------------
    # 3. INDIVIDUAL ADDITIVE ODDS CONTRIBUTIONS
    # -----------------------------------------------------

    previous_margin = base_value

    odds_contributions = []

    for shap_value in feature_shap:

        current_margin = previous_margin + shap_value

        # Good : Bad odds
        previous_odds = np.exp(-previous_margin)

        current_odds = np.exp(-current_margin)

        contribution = current_odds - previous_odds

        odds_contributions.append(contribution)

        previous_margin = current_margin

    odds_contributions = np.array(odds_contributions)

    # -----------------------------------------------------
    # 4. BASELINE SCORE
    # -----------------------------------------------------

    base_score = (
        BASE_SCORE
        - FACTOR * (np.log(BASE_ODDS) + base_value)
    )

    # -----------------------------------------------------
    # 5. INDIVIDUAL SCORE CONTRIBUTIONS
    # -----------------------------------------------------

    feature_scores = -feature_shap * FACTOR

    # -----------------------------------------------------
    # 6. FINAL SCORE
    # -----------------------------------------------------

    final_score = base_score + feature_scores.sum()

    # -----------------------------------------------------
    # 7. FINAL ODDS
    # -----------------------------------------------------

    calculated_odds = (
        base_odds + odds_contributions.sum()
    )

    # Final log-odds from SHAP
    total_margin = base_value + feature_shap.sum()

    # -----------------------------------------------------
    # 8. VALIDATION
    # -----------------------------------------------------

    # SHAP margin must reconcile with model probability
    shap_probability = 1 / (1 + np.exp(-total_margin))

    np.testing.assert_allclose(
        shap_probability,
        event_probability,
        rtol=1e-5,
        atol=1e-7,
        err_msg="SHAP probability does not match model probability"
    )

    # Additive odds must reconcile with model odds
    np.testing.assert_allclose(
        calculated_odds,
        model_odds,
        rtol=1e-5,
        atol=1e-7,
        err_msg="SHAP odds do not match model odds"
    )

    # -----------------------------------------------------
    # 9. FEATURE EXPLANATION DATAFRAME
    # -----------------------------------------------------

    explanation = pd.DataFrame({

        "Feature Name": input_df.columns,

        "SHAP Value": feature_shap,

        "Odds Contribution": odds_contributions,

        "Score Contribution": feature_scores

    })

    explanation = explanation.sort_values(
        by="Score Contribution"
    )

    # -----------------------------------------------------
    # 10. BASELINE AND TOTAL ROWS
    # -----------------------------------------------------

    baseline_row = pd.DataFrame([{

        "Feature Name": "Base Line",

        "SHAP Value": base_value,

        "Odds Contribution": base_odds,

        "Score Contribution": base_score

    }])

    total_row = pd.DataFrame([{

        "Feature Name": "Total",

        "SHAP Value": total_margin,

        "Odds Contribution": calculated_odds,

        "Score Contribution": final_score

    }])

    result_df = pd.concat(
        [
            baseline_row,
            explanation,
            total_row
        ],
        ignore_index=True
    )

    return {

        "probability": event_probability,

        "model_odds": model_odds,

        "base_value": base_value,

        "base_odds": base_odds,

        "base_score": base_score,

        "final_score": final_score,

        "final_odds": calculated_odds,

        "explanation": result_df

    }


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(result):

    final_score = result["final_score"]

    decision = "Approved" if final_score >= 500 else "Reject"

    decision_color = "#008000" if decision == "Approved" else "#C2183A"

    st.markdown(
        f"""
        <div style="
            background-color: {decision_color};
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-top: 10px;
        ">
            <h3 style="margin:0; color:white;">{decision}</h3>
            <p style="margin:5px 0 0 0;">Score: {final_score:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Model Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Bad/NPA Probability",
        f"{result['probability']:.2%}"
    )

    col2.metric(
        "Good : Bad Odds",
        f"{result['final_odds']:.4f}"
    )

    col3.metric(
        "Final Score",
        f"{result['final_score']:.2f}"
    )

    st.divider()

    st.subheader("SHAP Scorecard")

    st.dataframe(

        result["explanation"].style.format({

            "SHAP Value": "{:+.6f}",

            "Odds Contribution": "{:+.6f}",

            "Score Contribution": "{:+.2f}"

        }),

        use_container_width=True,

        hide_index=True

    )

    # Reconciliation
    st.divider()

    st.subheader("Reconciliation")

    odds_check = result["explanation"][
        "Odds Contribution"
    ].sum()

    score_check = result["explanation"][
        "Score Contribution"
    ].sum()

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Odds reconciliation**")

        st.write(
            f"Sum of contributions: {odds_check:.8f}"
        )

        st.write(
            f"Model odds: {result['model_odds']:.8f}"
        )

    with col2:

        st.write("**Score reconciliation**")

        st.write(
            f"Sum of contributions: {score_check:.4f}"
        )

        st.write(
            f"Final score: {result['final_score']:.4f}"
        )


# =========================================================
# MAIN STREAMLIT APPLICATION
# =========================================================

def main():

    st.set_page_config(
        page_title="SYM Scorecard",
        layout="wide"
    )

    st.title("Loan Application Scorecard")

    menu = [
        "About",
        "App"
    ]

    choice = st.sidebar.selectbox(
        "Menu",
        menu
    )

    if choice == "About":

        st.header("About the Application")

        st.write(
            """
            This application predicts loan application
            cancellation probability using an XGBoost model.

            SHAP is used to explain individual predictions.

            The application also converts SHAP contributions
            into additive odds contributions and score
            contributions.
            """
        )

    # -----------------------------------------------------
    # APP MENU
    # -----------------------------------------------------

    elif choice == "App":

        choice_2 = st.sidebar.selectbox(

            "Menu",

            [
                "New Application",
                "By Login IDs"
            ]

        )

        model = load_model()

        df = load_data()

        feature_names = model.get_booster().feature_names

        if not feature_names:

            st.error("Model feature names are unavailable.")

            return

        # ================================================
        # NEW APPLICATION
        # ================================================

        if choice_2 == "New Application":

            st.header("New Loan Application")

            # Default values from first available row
            # default_row = df.iloc[0]
            default_row = df[df['app_id'] == 18073].iloc[0]

            with st.form(key="new_application_form"):

                selected_features = {}

                st.write("Enter application details")

                for feature in feature_names:

                    default_value = default_row[feature]

                    if pd.isna(default_value):
                        default_value = 0.0

                    selected_features[feature] = st.number_input(

                        label=feature,

                        value=float(default_value),

                        key=f"new_{feature}"

                    )

                submit_button = st.form_submit_button(
                    "Calculate Score",
                    use_container_width=True
                )

            if submit_button:

                try:

                    input_df = pd.DataFrame(

                        [selected_features],

                        columns=feature_names

                    )

                    result = calculate_scorecard(
                        model,
                        input_df
                    )

                    display_results(result)

                except Exception as e:

                    st.error(f"Calculation failed: {e}")

                    st.code(traceback.format_exc())

        # ================================================
        # BY LOGIN IDS
        # ================================================

        elif choice_2 == "By Login IDs":

            st.header("Search by Application ID")

            if "app_id" not in df.columns:

                st.error("app_id column not found in dataset.")

                return

            with st.form(key="login_id_form"):

                appid = st.number_input(

                    "Enter Application ID",

                    min_value=10437,

                    step=1

                )

                submit_button = st.form_submit_button(

                    "Calculate Score",

                    use_container_width=True

                )

            if submit_button:

                try:

                    application_df = df.loc[
                        df["app_id"] == appid
                    ]

                    if application_df.empty:

                        st.warning(
                            f"No application found for ID: {appid}"
                        )

                        return

                    if len(application_df) > 1:

                        st.warning(
                            "Multiple records found. Using the first record."
                        )

                    input_df = application_df[
                        feature_names
                    ].iloc[[0]].copy()

                    result = calculate_scorecard(
                        model,
                        input_df
                    )

                    st.write(f"Application ID: {appid}")

                    display_results(result)

                except Exception as e:

                    st.error(f"Calculation failed: {e}")

                    st.code(traceback.format_exc())


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
