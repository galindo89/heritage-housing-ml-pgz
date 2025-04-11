# src/load_pipeline.py

import joblib
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