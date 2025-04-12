import streamlit as st
from src.load_model_data import load_pkl_files, load_data_sets

def model_performance_body():
    st.title("Model Performance Overview")

    st.header("1. Goal of the Model")
    st.markdown("""
    The model was developed to meet Business Requirement 2:  
    Predict the sale price of the 4 inherited houses and any other house in Ames, Iowa.

    The project set a performance target of achieving a minimum R² score of 0.75 on both the training and test datasets.
    """)
    st.header("2. Overview of Pipelines Used")
    st.markdown("""
    The final model was constructed by chaining together:
    - A **feature engineering pipeline** applied to the raw training data
    - A **modeling pipeline** that performs scaling, feature selection, and prediction using XGBoost
    
    XGBoost was selected as the final model after it demonstrated better generalization performance and stability compared to alternative algorithms such as Lasso and Random Forest.
    """)

    st.subheader("Model Pipeline")
    st.markdown("""
    This pipeline assumes the input data has already been preprocessed. It includes:
    - Scaling the engineered features using `StandardScaler`
    - Selecting a subset of features using `SelectFromModel`
    - Final regression using a tuned `XGBRegressor` model
    """)
    if st.checkbox("Show model pipeline structure"):
        model = load_pkl_files("outputs/ml_pipeline/saleprice/v1/lr_pipeline_model.pkl")
        st.write(model)

    st.subheader("Feature Engineering Pipeline")
    st.markdown("""
    This pipeline is resposible for data cleaning and feature engineering includes the following steps:

    - Dropping irrelevant features
    - Ordinal encoding of selected categorical variables
    - Median and mode imputation
    - Log10 transformation of skewed numerical features
    - Smart correlated feature filtering

    This pipeline was used during training and re-applied to test and live input data.
    """)
    if st.checkbox("Show feature engineering pipeline structure"):
        feature_pipeline = load_pkl_files("outputs/ml_pipeline/saleprice/v1/lr_pipeline_data_cleaning_feat_eng.pkl")
        st.write(feature_pipeline)
      
    st.header("3. Final Model Scores")
    st.markdown("""
    The final model achieved the following performance scores:

    - R² Score (Train): [insert value]
    - R² Score (Test): [insert value]

    These results met the project’s performance target and indicate good generalization.
    """)

    st.header("5. Feature Importance Summary")
    st.markdown("""
    The following table shows the importance values assigned by the trained model:

    | Feature        | Importance Score |
    |----------------|------------------|
    | OverallQual    | [value]          |
    | TotalBsmtSF    | [value]          |
    | GrLivArea      | [value]          |
    | YearBuilt      | [value]          |

    Use this table to summarize key predictors. Values should reflect your final model.
    """)

    st.header("6. Modeling with Reduced Features")
    st.markdown("""
    Experiments were conducted using only the most important features from correlation and model importance analysis.

    However, models trained with reduced features failed to meet the minimum R² threshold and were not selected for deployment.
    """)

    st.header("7. Why All Features Are Required")
    st.markdown("""
    The final pipeline performs multiple interdependent preprocessing steps.

    Features removed early may cause downstream issues in encoding, imputation, or transformation. Therefore, keeping the full feature set ensures the pipeline executes correctly and maintains performance.
    """)
