# src/load_pipeline.py

import joblib
import pandas as pd
import streamlit as st

# Load the trained pipeline or model from a specified path

@st.cache_data
def load_pkl_files(pipeline_path):
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"File not found: {pipeline_path}")
        return None    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None
    
# Function to load data sets for analysis and diagrams

@st.cache_data    
def  load_data_sets(data_set_path):
    try:
        data_set = pd.read_csv(data_set_path)
        return data_set
    except FileNotFoundError:
        st.error(f"File not found: {data_set_path}")
        return None    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None
    
  