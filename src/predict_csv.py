import pandas as pd
import streamlit as st

def predict_from_csv(csv_file, feature_pipeline, model):
    try:
        df_input = pd.read_csv(csv_file)

        required_columns = feature_pipeline.feature_names_in_
        missing_cols = set(required_columns) - set(df_input.columns)

        if missing_cols:
            return None, f"Missing columns in uploaded file: {missing_cols}", None

        # Transform and predict
        X_transformed = feature_pipeline.transform(df_input)
        log_preds = model.predict(X_transformed)
        saleprice_preds = (10 ** log_preds).round(2)

        df_output = df_input.copy()
        df_output["PredictedSalePrice"] = saleprice_preds

        return df_output, None, saleprice_preds.sum()

    except Exception as e:
        return None, f"Prediction error: {e}", None
