# import necessary packages
import streamlit as st

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
This interactive application demonstrates how different unsupervised machine learning methods work through several visualizations and explanations. \n\n In this app, you can:
- Navigate between **Principal Component Analysis**, **K-Means Clustering**, and **Hierarchical Clustering**;
- Upload your own dataset or choose one of the demo datasets available;
- Adjust your chosen model's hyperparameters and find the optimal ones;
- If your dataset has labeled data, gauge performance of clustering method.
### What is Unsupervised Machine Learning?
Unsupervised learning is a type of machine learning used on datasets with unlabeled data.
This means that the datasets contain only the features that will be used for prediction and not the target variable that will be predicted.
Unsupervised learning techniques may also be used on labeled data, especially for preprocessing. 

They identify patterns, relationships, and structures in the data own their own, rather than predicting outputs.
            """)