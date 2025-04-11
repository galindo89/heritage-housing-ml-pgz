import streamlit as st
from app_pages.multipage import MultiPage

# Import all pages scripts
from app_pages.home import home_page
from app_pages.house_price_predictions import prediction_page_body

# Create an instance of the MultiPage class
app = MultiPage("House Price Prediction App")

# Add pages to the app
app.add_page("Project Summary", home_page)
app.add_page("House Price Prediction", prediction_page_body)


# Run the app
app.run()