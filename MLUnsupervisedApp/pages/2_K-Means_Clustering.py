# import necessary packages
import streamlit as st

from Unsupervised_ML import *
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# -------------------------
# PAGE CONFIGURATIONS
# -------------------------
st.set_page_config(
    page_title="K-Means Clustering",
    page_icon=":material/graph_6:",
    layout="wide"
)

# -------------------------
# INITIAL PAGE STRUCTURE
# -------------------------

st.title(":material/graph_6: K-Means Clustering")

# -------------------------
# HELPER FUNCTIONS
# -------------------------

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
    is_labeled = st.sidebar.checkbox(label="Is the dataset labeled?")
    if is_labeled == True:
        target = st.sidebar.selectbox(label="What is the target variable?",
                                      options=dataset.columns,
                                      index=None)
    else:
        target = None

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

