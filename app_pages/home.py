import streamlit as st

def home_page():
    st.title("Inheritage Housing Price - Project Overview")

    st.markdown("## Project Purpose")
    st.write(
        """
        This project addresses a real estate challenge posed by a customer, who inherited four houses in Ames, Iowa.
        As the customer is unfamiliar with the local market, she seeks a data-driven solution to evaluate and maximize the sale price of her properties.
        """
    )

    st.markdown("## Business Requirements")
    st.write(
        """
        - Identify which house attributes correlate most strongly with sale price.
        - Predict the sale price of the four inherited houses and any other property in Ames, Iowa.
        """
    )

    st.markdown("## Dataset Summary")
    st.write(
        """
        The dataset, sourced from Kaggle, contains detailed attributes for 1,460 residential properties in Ames, Iowa.
        It includes both numerical and categorical features describing aspects such as size, location, condition, and amenities.
        The target variable is the house sale price.
        """
    )

    st.markdown("## Machine Learning Approach")
    st.write(
        """
        A regression model was developed to predict house sale prices using conventional machine learning techniques.
        The model was trained and validated on a cleaned and feature-engineered version of the dataset.
        An R² performance score of at least 0.75 on both the training and test sets was established as a success criterion.
        """
    )

    st.markdown("## App Navigation")
    st.write(
        """
        Use the sidebar to explore the following sections:
        - House Features Analysis
        - House Price Predictions
        - Hypothesis Validation
        - Model Performance
        """
    )