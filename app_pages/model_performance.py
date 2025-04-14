import streamlit as st
from src.load_model_data import load_pkl_files, load_data_sets
from src.model_performance_evaluator import evaluate_model_performance
from src.model_performance_evaluator import plot_feature_importance


def model_performance_body():
    """
    Display the model performance page of the Streamlit app.
    """
    st.title("Model Performance Overview")
    st.header("1. Goal of the Model")
    st.markdown("""
    The model was developed to meet Business Requirement 2:
    Predict the sale price of the 4 inherited houses and any
    other house in Ames, Iowa.

    During training, the target variable `SalePrice` was log-transformed using
    `log10` to reduce right skew and improve model fit.
    This transformation was reversed after prediction to obtain sale prices
    in their original scale.

    The project set a performance target of achieving a minimum
    R² score of 0.75 on both the training and test datasets.
    """)

    st.header("2. Overview of Pipelines Used")
    st.markdown("""
    The final model was constructed by chaining together:
    - A **feature engineering pipeline** applied to the raw training data
    - A **modeling pipeline** that performs scaling, feature selection,
    and prediction using XGBoost

    XGBoost was selected as the final model after it demonstrated better
    generalization performance and stability compared to alternative algorithms
    such as Lasso and Random Forest.
    """)

    st.subheader("Model Pipeline")
    st.markdown("""
    This pipeline assumes the input data has already been preprocessed.
    It includes:
    - Scaling the engineered features using `StandardScaler`
    - Selecting a subset of features using `SelectFromModel`
    - Final regression using a tuned `XGBRegressor` model
    """)
    if st.checkbox("Show model pipeline structure"):
        try:
            model = load_pkl_files("outputs/ml_pipeline/saleprice/v1/lr_pipeline_model.pkl")
            st.write(model)
        except FileNotFoundError:
            st.error("Could not find 'lr_pipeline_model.pkl' in outputs directory.")
        except Exception as e:
            st.error(f"An error occurred while loading the model pipeline: {e}")

    st.subheader("Feature Engineering Pipeline")
    st.markdown("""
    This pipeline is resposible for data cleaning and feature engineering
    includes the following steps:

    - Dropping irrelevant features
    - Ordinal encoding of selected categorical variables
    - Median and mode imputation
    - Log10 transformation of skewed numerical features
    - Smart correlated feature filtering

    This pipeline was used during training and re-applied to test and
    live input data.
    """)
    if st.checkbox("Show feature engineering pipeline structure"):
        try:
            feature_pipeline = load_pkl_files("outputs/ml_pipeline/saleprice/v1/lr_pipeline_data_cleaning_feat_eng.pkl")
            st.write(feature_pipeline)
        except FileNotFoundError:
            st.error("Could not find 'lr_pipeline_data_cleaning_feat_eng.pkl' in outputs directory.")
        except Exception as e:
            st.error(f"An error occurred while loading the feature engineering pipeline: {e}")

    st.header("3. Model Performance")
    st.markdown("""
    To evaluate model performance, we use three standard regression metrics:

    - **Mean Absolute Error (MAE)**: Measures the average absolute difference
    between actual and predicted values.
    - **Root Mean Squared Error (RMSE)**: Measures the square root of the
    average squared differences, giving more weight to larger errors.
    - **R² Score (Coefficient of Determination)**: Indicates how well the model
    explains the variance in the target variable.

    Since the target variable `SalePrice` was log-transformed during training,
    we evaluate model performance on both the **log scale** and the **original
    scale** (after inverse transformation). This allows us to assess both model
    fit and interpretability in actual currency values.
    """)

    X_train = load_data_sets("outputs/ml_pipeline/saleprice/v1/X_train.csv")
    Y_train = load_data_sets("outputs/ml_pipeline/saleprice/v1/y_train.csv")
    X_test = load_data_sets("outputs/ml_pipeline/saleprice/v1/X_test.csv")
    Y_test = load_data_sets("outputs/ml_pipeline/saleprice/v1/y_test.csv")
    Y_train_log = load_data_sets("outputs/ml_pipeline/saleprice/v1/y_train_log.csv")
    Y_test_log = load_data_sets("outputs/ml_pipeline/saleprice/v1/y_test_log.csv")
    pipeline_model = load_pkl_files("outputs/ml_pipeline/saleprice/v1/lr_pipeline_model.pkl")

    evaluate_model_performance(
    pipeline=pipeline_model,
    X_train=X_train,
    X_test=X_test,
    y_train_log=Y_train_log,
    y_test_log=Y_test_log,
    y_train=Y_train,
    y_test=Y_test
)
    st.header("4. Feature Importance Summary")
    st.markdown("""
    The following table shows the importance values assigned by the
    trained model:""")

    plot_feature_importance(pipeline_model, X_train)
    
    st.header("5. Modeling with Reduced Features")
    st.markdown("""
    Experiments were conducted using only the most important features from
    correlation and model importance analysis.

    However, models trained with reduced features failed to meet the minimum
    R² threshold and were not selected for deployment.
    """)

    st.header("6. Important Notes")
    st.markdown("""
    - The model was trained using all features available in the training
    dataset, including those with strong correlation to `SalePrice`.  
    As a result, the input data must include all expected variables
    to ensure correct pipeline execution and accurate predictions.  
    This may be revised in future versions with further feature reduction
    and pipeline refactoring.

    - The current design assumes the full feature set is passed through the
    feature engineering pipeline, which includes encoding, imputation,
    transformation, and correlation filtering. Skipping or manually removing
    features before prediction will result in pipeline errors or degraded performance.
    """)
