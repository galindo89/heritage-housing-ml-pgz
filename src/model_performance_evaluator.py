import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model_performance(pipeline, X_train, X_test, y_train_log, y_test_log, y_train, y_test):
    # Predictions in log scale
    y_train_pred_log = pipeline.predict(X_train)
    y_test_pred_log = pipeline.predict(X_test)

    st.subheader("Model Evaluation - Log Scale")

    col1, col2, col3 = st.columns(3)
    col1.metric("Train R² (log)", f"{r2_score(y_train_log, y_train_pred_log):.4f}")
    col2.metric("Train RMSE (log)", f"{mean_squared_error(y_train_log, y_train_pred_log, squared=False):.4f}")
    col3.metric("Train MAE (log)", f"{mean_absolute_error(y_train_log, y_train_pred_log):.4f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test R² (log)", f"{r2_score(y_test_log, y_test_pred_log):.4f}")
    col2.metric("Test RMSE (log)", f"{mean_squared_error(y_test_log, y_test_pred_log, squared=False):.4f}")
    col3.metric("Test MAE (log)", f"{mean_absolute_error(y_test_log, y_test_pred_log):.4f}")

    st.markdown("---")

    # Back-transform predictions to original scale
    y_train_pred = np.power(10, y_train_pred_log)
    y_test_pred = np.power(10, y_test_pred_log)

    st.subheader("Model Evaluation - Original Scale")

    col1, col2, col3 = st.columns(3)
    col1.metric("Train R²", f"{r2_score(y_train, y_train_pred):.4f}")
    col2.metric("Train RMSE", f"{mean_squared_error(y_train, y_train_pred, squared=False):,.2f}")
    col3.metric("Train MAE", f"{mean_absolute_error(y_train, y_train_pred):,.2f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test R²", f"{r2_score(y_test, y_test_pred):.4f}")
    col2.metric("Test RMSE", f"{mean_squared_error(y_test, y_test_pred, squared=False):,.2f}")
    col3.metric("Test MAE", f"{mean_absolute_error(y_test, y_test_pred):,.2f}")




def plot_feature_importance(pipeline, X_train, top_n=10, title="Feature Importances"):

    try:
        selector = pipeline.named_steps["feature_selection"]
        model = pipeline.named_steps["model"]

        if not hasattr(model, "feature_importances_"):
            st.warning("Model does not expose 'feature_importances_' attribute.")
            return

        selected_features = X_train.columns[selector.get_support()]
        importances = model.feature_importances_

        df_feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(top_n)

        st.subheader(title)
        st.dataframe(df_feature_importance)

        st.markdown("**Feature Importance Plot:**")        
        fig, ax = plt.subplots(figsize=(10, 5))
        df_feature_importance.head(top_n).plot(
            kind='bar',
            x='Feature',
            y='Importance',
            legend=False,
            ax=ax
        )
        ax.set_title("Feature Importances from Refined XGBoost Model")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error displaying feature importances: {e}")
