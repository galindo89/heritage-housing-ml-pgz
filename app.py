import streamlit as st
from app_pages.multipage import MultiPage
print  ("Starting the app...")

# Import all pages scripts
from app_pages.home import home_page

# Create an instance of the MultiPage class
app = MultiPage("House Price Prediction App")

# Add pages to the app
app.add_page("Project Summary", home_page)


# Run the app
app.run()