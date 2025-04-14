import streamlit as st
import pandas as pd
from src.load_model_data import load_data_sets
import matplotlib.pyplot as plt
import seaborn as sns


def feature_analysis_body():
    st.title("Exploratory Data Analysis")

    st.header("1. Business Context")
    st.markdown("""
    As part of the business requirement of this project, it was required to
    make an analysis of the house features to understand the most important
    ones to predict house prices.

    To better understand the house price distribution and the driving factors
    behind valuation, we examine key variables that influence `SalePrice`
    in Ames, Iowa.

    This analysis addresses **Business Requirement 1**:
    > *Identify which house attributes correlate most strongly with
    sale price.*
    """)

    st.header("2. Dataset Overview")
    try:
        df = load_data_sets("outputs/datasets/collection/house_prices_records.csv")
        st.write("**Dataset Dimensions:**")
        st.write(f"- Entries (rows): {df.shape[0]}")
        st.write(f"- Features (columns): {df.shape[1]}")
        if st.checkbox("Show sample data"):
            st.dataframe(df.head())
    except FileNotFoundError:
        st.error("Could not find 'house_prices_records.csv' in outputs directory.")

    st.header("3. Handling of Categorical Data")
    st.markdown("""
    The dataset includes multiple categorical features such as `KitchenQual`,
    `GarageFinish`, `BsmtExposure`, and `BsmtFinType1`. We explored their
    meaning and used boxplots to understand their relationship with the target
    variable `SalePrice`. These plots clearly showed that these variables
    follow an ordinal pattern — for instance, `KitchenQual` ranges from "Fair"
    to "Excellent", and `GarageFinish` from "Unfinished" to "Finished".

    See boxplots below showing the relationship between these categorical
    features and `SalePrice`:
    """)

    try:
        cat_cols = ['BsmtExposure', 'BsmtFinType1', 'KitchenQual', 'GarageFinish']
        df = load_data_sets("outputs/datasets/collection/house_prices_records.csv")

        if st.checkbox("Show boxplots for categorical features"):
            st.write("**Boxplots of categorical features against SalePrice:**")
            for col in cat_cols:
                fig, ax = plt.subplots(figsize=(8, 5))
                df.boxplot(column='SalePrice', by=col, ax=ax)
                ax.set_title(f'SalePrice by {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('SalePrice')
                plt.suptitle('')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

    except FileNotFoundError:
        st.error("Could not find 'house_prices_records.csv' in outputs directory.")
    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")

    st.markdown("""
    To evaluate the correlation towards SalePrice using Spearman, we first 
    needed to **ordinally encode** them, assigning a logical numeric order
    to each level:

    | Variable         | Encoding Order              |
    |------------------|-----------------------------|
    | KitchenQual      | Fa < TA < Gd < Ex           |
    | GarageFinish     | Unf < RFn < Fin             |
    | BsmtExposure     | No < Mn < Av < Gd           |
    | BsmtFinType1     | Unf < Rec < BLQ < ALQ < GLQ |

    Encoding them this way preserves their **ordinal nature** and allows us to
    measure correlation accurately.
    To differentiate between the original and encoded features, we added
    `_Enc` to the encoded feature names.
    """)

    st.header("4. Features Most Correlated with SalePrice")
    st.markdown("""
    Based on both **Pearson** and **Spearman** correlation analyses, 
    the following features showed strong relationships with `SalePrice`:

    | Feature            | Pearson | Spearman |
    |--------------------|---------|----------|
    | OverallQual        | 0.79    | 0.81     |
    | GrLivArea          | 0.71    | 0.73     |
    | GarageArea         | 0.64    | 0.70     |
    | TotalBsmtSF        | 0.62    | 0.67     |
    | 1stFlrSF           | 0.61    | 0.67     |
    | YearBuilt          | 0.59    | 0.68     |
    | YearRemodAdd       | 0.53    | 0.58     |
    | KitchenQual_Enc    | 0.67    | 0.69     |
    | GarageFinish_Enc   | 0.65    | 0.66     |
    | GarageYrBlt        | 0.55    | 0.62     |

    """)

    if st.checkbox("Show correlations ≥ 0.5 (Pearson vs Spearman)"):
        try:
            df = load_data_sets("outputs/datasets/collection/house_prices_records_encoded.csv")
            numeric_df = df.select_dtypes(include='number')


            pearson_corr = numeric_df.corr(method='pearson')['SalePrice'].drop('SalePrice')
            pearson_high = pearson_corr[pearson_corr.abs() >= 0.5].index.tolist()

            spearman_corr = numeric_df.corr(method='spearman')['SalePrice'].drop('SalePrice')
            spearman_high = spearman_corr[spearman_corr.abs() >= 0.5].index.tolist()

            pearson_features = list(set(pearson_high + ['SalePrice']))
            spearman_features = list(set(spearman_high + ['SalePrice']))

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            sns.heatmap(
                numeric_df[pearson_features].corr(method='pearson'),
                annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0], cbar=False
            )
            axes[0].set_title("Pearson Correlations ≥ 0.5")

            
            sns.heatmap(
                numeric_df[spearman_features].corr(method='spearman'),
                annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1], cbar=False
            )
            axes[1].set_title("Spearman Correlations ≥ 0.5")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except FileNotFoundError:
            st.error("Could not find 'house_prices_records_encoded.csv' in outputs directory.")
        except Exception as e:
            st.error(f"An error occurred while plotting: {e}")


    if st.checkbox("Show scatter plots for selected features"):
        try:
            df= load_data_sets("outputs/datasets/collection/house_prices_records_encoded.csv")
            selected_features = [
            'OverallQual', 'GrLivArea', 'KitchenQual_Enc', 'YearBuilt',
            'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'GarageFinish_Enc',
            'GarageYrBlt', 'YearRemodAdd'
            ]

            for col in selected_features:
                fig, ax = plt.subplots(figsize=(12, 4))                        
                sns.scatterplot(data=df, x=col, y='SalePrice', ax=ax)
                ax.set_title(f'{col} vs SalePrice')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        except FileNotFoundError:
            st.error("Could not find 'house_prices_records_encoded.csv' in outputs directory.")
        except Exception as e:
            st.error(f"An error occurred while plotting: {e}")
