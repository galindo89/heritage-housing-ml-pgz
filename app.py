import streamlit as st
from app_pages.multipage import MultiPage

# Import all pages scripts
from app_pages.home import home_page
from app_pages.house_price_predictions import prediction_page_body
from app_pages.house_features_analysis import feature_analysis_body
from app_pages.hypothesis import hypothesis_page_body
from app_pages.model_performance import model_performance_body

# Create an instance of the MultiPage class
app = MultiPage("House Price Prediction App")

# Add pages to the app
app.add_page("Project Summary", home_page)
app.add_page("House Features Analysis", feature_analysis_body)
app.add_page("House Price Prediction", prediction_page_body)
app.add_page("Hypothesis Validation", hypothesis_page_body)
app.add_page("Model Performance Overview", model_performance_body)

# Run the app
app.run()