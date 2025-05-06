# import necessary packages
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------
# PAGE CONFIGURATIONS
# -------------------------
st.set_page_config(
    page_title="Unsupervised Machine Learning",
    page_icon=":house:",
    layout="wide"
)

# -------------------------
# MAIN PAGE STRUCTURE
# -------------------------
st.title("Unsupervised Machine Learning Explorer")
st.markdown("""
### About this Application
This interactive application demonstrates how different unsupervised machine learning methods work through several visualizations and explanations. You can:
- Navigate between **Principal Component Analysis**, **K-Means Clustering**, and **Hierarchical Clustering**;
- Upload your own dataset or choose one of the demo datasets available;
- Adjust your chosen model's hyperparameters and find the optimal ones;
- If your dataset has labeled data, gauge performance of clustering method.
### What is Unsupervised Machine Learning?

            """)

# --------------------------
# HELPER FUNCTIONS
# --------------------------

# turn sklearn toy datasets into pandas dataframes
def toy_to_df(load_function):
    bunch = load_function()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df

# data preprocessing
def data_preprocessing(df, target):
    # drop rows with missing data
    df = df.dropna()
    # encode categorical variables
    categorical_cols = [col for col in df.columns
                        if (pd.api.types.is_categorical_dtype(df[col].dtype)
                        or pd.api.types.is_object_dtype(df[col].dtype))
                        and col != target]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # define feature and target variables
    if target is not None:
        X = df.drop(target, axis=1)
        y = df[target]
    else:
        X = df
        y = None
    # center and scale features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return df, X_std, y

# principal component analysis
def data_pca(X_std, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    explained_variance = pca.explained_variance_ratio_
    exp_variance_cumsum = np.cumsum(explained_variance)
    return X_pca, exp_variance_cumsum
