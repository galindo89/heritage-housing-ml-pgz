import streamlit as st

def hypothesis_page_body():
    """
    Display the hypothesis validation page of the Streamlit app.
    """
    st.title("Hypothesis Validation")

    st.header("1. Business Hypotheses")
    st.markdown("""
    At the beginning of the project, several hypotheses were formulated based on domain knowledge and exploratory analysis. The aim was to understand which house features have a significant impact on `SalePrice` and whether these patterns would hold through model training and validation.

    These hypotheses aligned with Business Requirement 1:
    Identify which house attributes correlate most strongly with sale price.
    """)

    st.header("2. Hypotheses and Validation Outcomes")
    st.markdown("""
    | Hypothesis | Outcome | Notes |
    |-----------|---------|-------|
    | KitchenQual influences SalePrice positively | Partially supported | Initially confirmed through boxplots and correlation; dropped due to high correlation with OverallQual |
    | GarageFinish influences SalePrice positively | Partially supported | Ordinal trend observed, but removed due to multicollinearity |
    | GrLivArea is a strong predictor | Confirmed | Retained and scored approximately 16 percent in feature importance |
    | 1stFlrSF contributes significantly | Rejected | Dropped due to multicollinearity during correlation filtering |
    | YearBuilt affects SalePrice | Confirmed | Ranked fourth in model importance |
    | GarageYrBlt improves prediction | Rejected | Dropped due to correlation with YearBuilt |
    | TotalBsmtSF is predictive | Confirmed | Second highest feature importance in final model |
    | YearRemodAdd adds value | Rejected | Removed during correlation filtering |
    """)

    st.header("3. Model-Based Feature Importance")
    st.markdown("""
    The final model, after correlation filtering and feature engineering, assigned the following relative importance to features:

    | Feature        | Importance Score |
    |----------------|------------------|
    | OverallQual    | 0.4989           |
    | TotalBsmtSF    | 0.2003           |
    | GrLivArea      | 0.1583           |
    | YearBuilt      | 0.1426           |
    """)

    st.header("4. Observations")
    st.markdown("""
    Features with strong initial correlation to SalePrice were not always retained in the final model due to redundancy or high inter-correlation. The use of SmartCorrelatedSelection helped ensure that only one representative variable from a group of highly correlated features was included.

    The final model relies on features that are both statistically sound and predictively reliable.
    """)
    
