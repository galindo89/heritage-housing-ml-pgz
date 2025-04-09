import streamlit as st
from app_pages.multipage import MultiPage
print  ("Starting the app...")

# Import all pages scripts
from app_pages.page_summary import page_summary_body

# Create an instance of the MultiPage class
app = MultiPage("Inherited House Price Prediction App")

# Add pages to the app
app.add_page("Project Summary", page_summary_body)


# Run the app
app.run()