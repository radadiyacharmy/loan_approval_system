import streamlit as st 
import requests
from streamlit_lottie import st_lottie
import shap
import numpy as np
import pandas as pd
# from src.pipelines.prediction_pipeline import Prediction
# from src.components.model_trainning import ModelTrainning
# from src.utils import load_obj


# 'state', 'area_code', 'international_plan', 'voice_mail_plan',
#        'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
#        'total_day_charge', 'total_eve_charge', 'total_night_charge',
#        'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
#        'number_customer_service_calls'

# state                                       NH
# area_code                        area_code_415
# international_plan                          no
# voice_mail_plan                             no
# number_vmail_messages                        0
# total_day_minutes                        254.3
# total_day_calls                            113
# total_day_charge                         43.23
# total_eve_charge                          6.71
# total_night_charge                        6.89
# total_intl_minutes                        11.8
# total_intl_calls                             2
# total_intl_charge                         3.19
# number_customer_service_calls                2


def load_lottie_url(url):

    r = requests.get(url)
    
    if r.status_code != 200:
        return None
    else:
        return r.json()

def main():

    st.set_page_config(page_title='loan_application', layout='wide')

    # lottie_file = load_lottie_url('https://lottie.host/598e9e70-6768-410d-b309-b8e8c6be56a2/N9wcpBAsqi.json')
    lottie_file = load_lottie_url('https://lottie.host/59ca70ea-e33d-453f-968e-26d42466abee/q2wjnXKxyV.json')

    menu = ['About', 'App']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'About':

        with st.container():

            st.write('To run the application, select app from Menu dropdown from navigation bar present on left side')

        with st.container():
            st.write('---')
            left_col, right_col = st.columns(2)

            with left_col:
                
                st.header('Problem Statment')
                st.write('##')
                st.write(
                    """
                            The problem at hand revolves around predicting customer churn, a crucial task in various industries, particularly in subscription-based services such as telecommunications, banking, and SaaS companies. Customer churn refers to the phenomenon where customers discontinue their services or subscriptions with a company. Identifying potential churners beforehand allows businesses to take proactive measures such as targeted marketing campaigns or personalized retention strategies to mitigate churn and retain valuable customers.
                    """
                )
            
            with right_col:
                st_lottie(lottie_file, height=300, key='customer_retention')

        with st.container():

            git_hub_url = 'https://github.com/soil15/Customer_churn'
            # model_tran = ModelTrainning()
            # class_repo = load_obj(model_tran.model_trainning_cofg_obj.class_repo_path)

            st.write('---')
            st.header('Key Steps in Solving the Problem:')
            st.markdown("[Github Repository](%s)" % git_hub_url)
            st.write('##')

            # ---------------------------------------
            st.subheader('1. Data Ingestion')
            st.write('I began by ingesting the raw data using a robust data ingestion pipeline, ensuring that data from various sources is seamlessly integrated for further processing.')

            # ---------------------------------------
            st.subheader('2. Data Transformation')
            st.write('Next, I focused on transforming the raw data into a format suitable for model training. This involved preprocessing steps such as handling missing values, encoding categorical variables, and splitting the data into training and testing sets.')

            # ---------------------------------------
            st.subheader('3. Model Training')
            st.write('Leveraging the XGBoost classifier and the optimized hyperparameters obtained through tuning, I trained a predictive model on the preprocessed data. This model is finely tuned to prioritize recall, ensuring that it effectively identifies potential churners, which is crucial for implementing targeted retention strategies.')

            # ---------------------------------------
            st.subheader('4. Hyperparameter Tuning for Recall Optimization')
            st.write('To further enhance model performance, I employed advanced techniques such as hyperparameter tuning with a primary focus on maximizing the recall score. By defining a custom scoring function that prioritizes precision, and conducting grid search cross-validation over a range of hyperparameters, I fine-tuned the XGBoost classifier to effectively identify potential churners while maintaining reasonable precision.')
            
            # ---------------------------------------
            st.subheader('5. Evaluation and Model Deployment')
            st.write(' After training the model, I thoroughly evaluated its performance using metrics such as confusion matrix, classification report.')
            # st.write(class_repo)


    elif choice == 'App':

        menu_2 = ['by_app_id', 'by_input']
        choice_2 = st.sidebar.selectbox('Menu', menu_2)


        if choice_2 == 'by_input':


            with st.form(key='input_form'):

                appid = st.number_input('Enter application ID', min_value=10437)

                submit_button = st.form_submit_button()

                model = pd.read_pickle(r'C:\Users\sahil\OneDrive\Desktop\SYM_SCORECARD\results\model\model_27_11_2026_02_04_44\model.pkl')

                df = pd.read_csv(r'C:\Users\sahil\OneDrive\Desktop\SYM_SCORECARD\base_data\home_loan_applications_features_20k_extended.csv')

                df = df.head(1)

                selected_features = {}

                for feature in model.get_booster().feature_names:
                    selected_features[feature] = st.number_input(
                        label=feature,
                        value=float(df[feature].values[0]),
                        key=feature
                    )   

                input_df = pd.DataFrame(
                        [selected_features],
                        columns=model.get_booster().feature_names
                    )

                if submit_button:
                        
                    df = df[df['app_id'] == appid]

                    model = pd.read_pickle(r'C:\Users\sahil\OneDrive\Desktop\SYM_SCORECARD\results\model\model_27_11_2026_01_05_07\model.pkl')
                    
                    input_df = df[model.get_booster().feature_names]

                    # Scorecard scaling parameters
                    BASE_SCORE = 600
                    BASE_ODDS = 50
                    PDO = 20

                    FACTOR = PDO / np.log(2)


                    # Initialize once when your API starts, not for every request
                    explainer = shap.TreeExplainer(
                        model,
                        model_output="raw"
                    )

                    # Calculate SHAP values for the incoming applicant
                    shap_values = explainer(input_df)

                    # Extract feature contributions
                    feature_shap = shap_values.values[0]

                    # Extract SHAP base value
                    base_value = float(
                        np.asarray(shap_values.base_values).reshape(-1)[0]
                    )

                    # Convert base value into base score
                    base_score = (
                        BASE_SCORE
                        - FACTOR * (np.log(BASE_ODDS) + base_value)
                    )

                    # Convert individual SHAP values into score points
                    feature_scores = -feature_shap * FACTOR

                    # Final score
                    final_score = base_score + feature_scores.sum()

                    # Feature-wise explanation
                    explanation = pd.DataFrame({
                        "feature": input_df.columns,
                        "shap_value": feature_shap,
                        "score_contribution": feature_scores
                    })

                    explanation = explanation.sort_values(
                        by="score_contribution"
                    )

                    result = {
                        "base_value": base_value,
                        "base_score": round(float(base_score), 2),
                        "feature_contributions": explanation,
                        "final_score": round(float(final_score), 2)
                    }
                    
                    # Extract results
                    base_value = result["base_value"]
                    base_score = result["base_score"]
                    feature_df = result["feature_contributions"]
                    final_score = result["final_score"]

                    # Baseline odds (Good:Bad)
                    base_odds = np.exp(-base_value)

                    # Prepare display dataframe
                    display_df = pd.DataFrame({
                        "Feature Name": feature_df["feature"],
                        "Odds": np.exp(feature_df["shap_value"]),
                        "Score": feature_df["score_contribution"]
                    })

                    # Baseline row
                    baseline_row = pd.DataFrame({
                        "Feature Name": ["Base Line"],
                        "Odds": [base_odds],
                        "Score": [base_score]
                    })

                    # Total odds
                    total_odds = base_odds * np.exp(
                        -feature_df["shap_value"].sum()
                    )

                    # Total row
                    total_row = pd.DataFrame({
                        "Feature Name": ["Total"],
                        "Odds": [total_odds],
                        "Score": [final_score]
                    })

                    # Combine rows
                    display_df = pd.concat(
                        [baseline_row, display_df, total_row],
                        ignore_index=True
                    )

                    # Display
                    st.dataframe(
                        display_df.style.format({
                            "Odds": "{:.2f}",
                            "Score": "{:+.2f}"
                        }),     
                        use_container_width=True,
                        hide_index=True
                    )
                    

        elif choice_2 == 'by_app_id':

            with st.form(key='input_form'):

                appid = st.number_input('Enter application ID', min_value=10437)

                submit_button = st.form_submit_button()
                
                if submit_button:

                        
                    df = pd.read_csv(r'C:\Users\sahil\OneDrive\Desktop\SYM_SCORECARD\base_data\home_loan_applications_features_20k_extended.csv')

                    df = df[df['app_id'] == appid]

                    model = pd.read_pickle(r'C:\Users\sahil\OneDrive\Desktop\SYM_SCORECARD\results\model\model_27_11_2026_01_05_07\model.pkl')
                    
                    input_df = df[model.get_booster().feature_names]

                    # Scorecard scaling parameters
                    BASE_SCORE = 600
                    BASE_ODDS = 50
                    PDO = 20

                    FACTOR = PDO / np.log(2)


                    # Initialize once when your API starts, not for every request
                    explainer = shap.TreeExplainer(
                        model,
                        model_output="raw"
                    )

                    # Calculate SHAP values for the incoming applicant
                    shap_values = explainer(input_df)

                    # Extract feature contributions
                    feature_shap = shap_values.values[0]

                    # Extract SHAP base value
                    base_value = float(
                        np.asarray(shap_values.base_values).reshape(-1)[0]
                    )

                    # Convert base value into base score
                    base_score = (
                        BASE_SCORE
                        - FACTOR * (np.log(BASE_ODDS) + base_value)
                    )

                    # Convert individual SHAP values into score points
                    feature_scores = -feature_shap * FACTOR

                    # Final score
                    final_score = base_score + feature_scores.sum()

                    # Feature-wise explanation
                    explanation = pd.DataFrame({
                        "feature": input_df.columns,
                        "shap_value": feature_shap,
                        "score_contribution": feature_scores
                    })

                    explanation = explanation.sort_values(
                        by="score_contribution"
                    )

                    result = {
                        "base_value": base_value,
                        "base_score": round(float(base_score), 2),
                        "feature_contributions": explanation,
                        "final_score": round(float(final_score), 2)
                    }
                    
                    # Extract results
                    base_value = result["base_value"]
                    base_score = result["base_score"]
                    feature_df = result["feature_contributions"]
                    final_score = result["final_score"]

                    # Baseline odds (Good:Bad)
                    base_odds = np.exp(-base_value)

                    # Prepare display dataframe
                    display_df = pd.DataFrame({
                        "Feature Name": feature_df["feature"],
                        "Odds": np.exp(feature_df["shap_value"]),
                        "Score": feature_df["score_contribution"]
                    })

                    # Baseline row
                    baseline_row = pd.DataFrame({
                        "Feature Name": ["Base Line"],
                        "Odds": [base_odds],
                        "Score": [base_score]
                    })

                    # Total odds
                    total_odds = base_odds * np.exp(
                        -feature_df["shap_value"].sum()
                    )

                    # Total row
                    total_row = pd.DataFrame({
                        "Feature Name": ["Total"],
                        "Odds": [total_odds],
                        "Score": [final_score]
                    })

                    # Combine rows
                    display_df = pd.concat(
                        [baseline_row, display_df, total_row],
                        ignore_index=True
                    )

                    # Display
                    st.dataframe(
                        display_df.style.format({
                            "Odds": "{:.2f}",
                            "Score": "{:+.2f}"
                        }),     
                        use_container_width=True,
                        hide_index=True
                    )
                    

                    # pred_obj = Prediction(state, area_code, inter_plan, voice_mail_plan, num_vmail_msg, total_day_min, total_day_calls, total_day_charge,
                    #             total_eve_charge, total_night_charge, total_int_min, total_int_calls, total_int_charge,
                    #             number_cust_srv_calls)

                    # data_df = pred_obj.get_data()

                    # transformed_df, pred, _ = pred_obj.get_predictions()

                    # if pred == 1:
                    #     st.success('Customer will churn')
                    # elif pred == 0:
                    #     st.success('Customer will not churn')



if __name__ == '__main__':

    main()