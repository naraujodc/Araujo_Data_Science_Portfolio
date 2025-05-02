# import necessary packages
import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# -------------------------
# PAGE CONFIGURATIONS
# -------------------------
st.set_page_config(
    page_title="Principal Component Analysis",
    page_icon=":material/graph_3:",
    layout="wide"
)

# -------------------------
# INITIAL PAGE STRUCTURE
# -------------------------

# -------------------------
# HELPER FUNCTIONS
# -------------------------
# turn sklearn toy datasets into pandas dataframes
def toy_to_df(load_function):
    bunch = load_function()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df

# -------------------------
# SIDEBAR STRUCTURE
# -------------------------
# dataset upload option
st.sidebar.markdown("## Dataset Selection")
dataset_upload = st.sidebar.file_uploader(label="Upload your own dataset",
                                          type="csv")

# use uploaded dataset if user inputs one
if dataset_upload is not None:
    dataset = pd.read_csv(dataset_upload)
    target = st.sidebar.selectbox(label="What is the target variable?",
                                  options=dataset.columns,
                                  index=None)

# dataset demo option
dataset_demo = None
if dataset_upload is None:
    st.sidebar.markdown("#### No dataset? Use a demo")
    dataset_demo = st.sidebar.selectbox(label="Demo datasets",
                                        options=["Breast Cancer", "Iris", "Wine"],
                                        index=None)

# use demo datasets if user chooses one
if dataset_demo == "Breast Cancer":
    dataset = toy_to_df(load_breast_cancer)
    target = "target"
elif dataset_demo == "Iris":
    dataset = toy_to_df(load_iris)
    target = "target"
elif dataset_demo == "Wine":
    dataset = toy_to_df(load_wine)
    target = "target"

# -------------------------
# MAIN PAGE STRUCTURE
# -------------------------