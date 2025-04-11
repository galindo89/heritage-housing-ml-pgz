import streamlit as st
import pandas as pd
from src.load_model_data import load_pkl_files
from src.predict_csv import predict_from_csv

VERSION = "v1"
PIPELINES_PATH = f"outputs/ml_pipeline/saleprice/{VERSION}/"
MODEL_NAME = "lr_pipeline_model.pkl"
FEATURE_ENG_NAME = "lr_pipeline_data_cleaning_feat_eng.pkl"
METADATA_PATH = "inputs/datasets/raw/house-metadata.txt"


def prediction_page_body():
    st.title("House Price Prediction")
    # Load both components
    feature_pipeline = load_pkl_files(f"{PIPELINES_PATH}{FEATURE_ENG_NAME}")
    model = load_pkl_files(f"{PIPELINES_PATH}{MODEL_NAME}")
    
    if feature_pipeline is None or model is None:
        st.warning("Could not load pipeline or model. Please check file paths.")
        return
    
    st.markdown("---")
    st.subheader("Predict from CSV")

    csv_file = st.file_uploader("Upload a CSV file with house attributes", type=["csv"])

    if csv_file is not None:
        df_output, error_msg, total_value = predict_from_csv(csv_file, feature_pipeline, model)

        if error_msg:
            st.error(error_msg)
        else:
            st.success("Predictions completed successfully.")
            st.dataframe(df_output)
            st.markdown(f"**Total predicted value for all properties:** ${total_value:,.2f}")
            st.download_button("Download predictions as CSV", df_output.to_csv(index=False), file_name="predicted_prices.csv")
    
    st.markdown("---")
    st.subheader("Predict from Direct Input")

    st.markdown("Use the form below to input house attributes and predict the sale price.")
    
    with st.expander("Click to view feature descriptions"):
        with open(METADATA_PATH, "r") as f:
            st.text(f.read())

    with st.form("prediction_form"):
        st.subheader("Input House Attributes")
        col1, col2 = st.columns(2)

        with col1:
            first_flr_sf = st.number_input("1st Floor SF", min_value=100, max_value=3000, value=1200)
            second_flr_sf = st.number_input("2nd Floor SF", min_value=0, max_value=3000, value=500)
            bedroom_abv_gr = st.selectbox("Bedrooms Above Ground", [0, 1, 2, 3, 4, 5])
            bsmt_exposure = st.selectbox("Basement Exposure", ["Gd", "Av", "Mn", "No", "None"])
            bsmt_fin_sf1 = st.number_input("Basement Finished SF1", min_value=0, max_value=2000, value=400)
            bsmt_fin_type1 = st.selectbox("Basement Finish Type 1", ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"])
            bsmt_unf_sf = st.number_input("Unfinished Basement SF", min_value=0, max_value=2000, value=300)
            enclosed_porch = st.number_input("Enclosed Porch SF", min_value=0, max_value=500, value=0)
            garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=500)
            garage_finish = st.selectbox("Garage Finish", ["Fin", "RFn", "Unf", "None"])
            garage_yr_blt = st.number_input("Garage Year Built", min_value=1900, max_value=2025, value=2000)

        with col2:
            gr_liv_area = st.number_input("Above Grade Living Area", min_value=300, max_value=4000, value=1500)
            kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
            lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=9000)
            lot_frontage = st.number_input("Lot Frontage (ft)", min_value=20.0, max_value=200.0, value=70.0)
            mas_vnr_area = st.number_input("Masonry Veneer Area", min_value=0, max_value=1000, value=100)
            open_porch_sf = st.number_input("Open Porch SF", min_value=0, max_value=500, value=50)
            overall_cond = st.selectbox("Overall Condition (1=worst, 10=best)", list(range(1, 11)), index=5)
            overall_qual = st.selectbox("Overall Quality (1=worst, 10=best)", list(range(1, 11)), index=5)
            total_bsmt_sf = st.number_input("Total Basement SF", min_value=0, max_value=3000, value=1000)
            wood_deck_sf = st.number_input("Wood Deck SF", min_value=0, max_value=1000, value=0)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1975)
            year_remod_add = st.number_input("Year Remodeled", min_value=1900, max_value=2025, value=2005)

        submit = st.form_submit_button("Predict Sale Price")

    if submit:
        input_dict = {
            '1stFlrSF': first_flr_sf,
            '2ndFlrSF': second_flr_sf,
            'BedroomAbvGr': bedroom_abv_gr,
            'BsmtExposure': bsmt_exposure,
            'BsmtFinSF1': bsmt_fin_sf1,
            'BsmtFinType1': bsmt_fin_type1,
            'BsmtUnfSF': bsmt_unf_sf,
            'EnclosedPorch': enclosed_porch,
            'GarageArea': garage_area,
            'GarageFinish': garage_finish,
            'GarageYrBlt': garage_yr_blt,
            'GrLivArea': gr_liv_area,
            'KitchenQual': kitchen_qual,
            'LotArea': lot_area,
            'LotFrontage': lot_frontage,
            'MasVnrArea': mas_vnr_area,
            'OpenPorchSF': open_porch_sf,
            'OverallCond': overall_cond,
            'OverallQual': overall_qual,
            'TotalBsmtSF': total_bsmt_sf,
            'WoodDeckSF': wood_deck_sf,
            'YearBuilt': year_built,
            'YearRemodAdd': year_remod_add
        }

        input_df = pd.DataFrame([input_dict])

        try:
            X_transformed = feature_pipeline.transform(input_df)
            prediction = model.predict(X_transformed)[0]
            predicted_price = round(10 ** prediction, 2)
            st.success(f"Predicted Sale Price: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")