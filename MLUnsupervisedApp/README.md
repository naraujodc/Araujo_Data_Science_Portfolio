# Unsupervised ML Explorer App
![Python](https://img.shields.io/badge/-Python-ffe873?style=flat&logo=python)&nbsp;
![Streamlit](https://img.shields.io/badge/Streamlit-ececec?style=flat&logo=streamlit)&nbsp;
![Scikitlearn](https://img.shields.io/badge/scikit_learn-101e27?logo=scikitlearn)&nbsp;
![SciPy](https://img.shields.io/badge/SciPy-575d63?logo=scipy)&nbsp;
![NumPy](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=flat&logo=numpy&logoColor=white)&nbsp;
![Pandas](https://img.shields.io/badge/pandas%20-%23150458.svg?&style=flat&logo=pandas&logoColor=white)&nbsp;
![Seaborn](https://img.shields.io/badge/Seaborn-79b6bc)&nbsp;
![Matplotlib](https://img.shields.io/badge/Matplotlib-1e3f5a)&nbsp;

## Table of Contents
[1. Project Overview (With Visual Examples)](#project-overview-with-visual-examples)\
[2. Instructions for Use](#instructions-for-use)\
[3. Files](#files)\
[4. Datasets Description](#datasets-description)\
[5. References](#references)

## Project Overview (With Visual Examples)
<p align="center">
<a href="https://unsupervised-ml-explorer.streamlit.app"><img alt="Unsupervised ML Explorer App"
src="https://img.shields.io/badge/Unsupervised_ML_Explorer_App-73a3bd?style=for-the-badge"/></a> &nbsp;
</p>

This project is an interactive app that invites the user to explore three different unsupervised machine learning methods:
**Principal Component Analysis** for dimensionality reduction, **K-Means Clustering**, and **Hierarchical Clustering**.
The user is able to:
- Upload their own dataset or choose from three demo datasets;
- Navigate between the three unsupervised ML models;
- Select different hyperparameters for each model, such as the k number of clusters and linkage method;
- Visualize the model outputs in different ways, including **scatterplots**, **biplots**, **scree plots**, **dendrograms**, and more;
- Read explanations about each aspect of the models, including the hyperparameters, metrics, and visuals shown, as well as the data preprocessing steps.
- View the first observations of their chosen dataset and its summary statistics, as well as short descriptions of the demo datasets.

The purpose of this project is to provide an interactive experience and enhance the user's understanding of unsupervised machine learning models.
My goal was to make the app as self-contained as possible, so there are answers to common questions that may show up and
explanations about the model outputs and visualizations within the app. This project showcases my knowledge of unsupervised machine learning
methods for dimensionality reduction and clustering and demonstrates my ability to interpret outputs and tune hyperparameters.

### Behind the App
Many computations happen in the background to ensure that the users can navigate the app smoothly and without error.
I highly recommend checking the code in this repository to see some of the strategies I used to guarantee that the code runs
without errors independently of the order in which users interact with the buttons.
<table style="border-collapse: collapse;">
  <tr style="border: none;">
    <td style="border: none;">
      <img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/MLUnsupervisedApp/images/uml-app-landing-page.png" alt="Landing page when user opens the app" width="100%"/>
      <br>
      <b>Landing page when user opens the app</b>
    </td>
    <td style="border: none;">
      <img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/MLUnsupervisedApp/images/uml-app-pca-page.png" alt="Sample navigation of PCA page" width="100%"/>
      <br>
      <b>Sample navigation of PCA page</b>
    </td>
  </tr>
  <tr style="border: none;">
    <td style="border: 0;">
      <img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/MLUnsupervisedApp/images/uml-app-kmeans-page.png" alt="Sample nagivation of K-Means page" width="100%"/>
      <br>
      <b>Sample navigation of K-Means Clustering page</b>
    </td>
    <td style="border: none;">
      <img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/MLUnsupervisedApp/images/uml-app-hier-page.png" alt="Sample navigation of Hierarchical page" width="100%"/>
      <br>
      <b>Sample navigation of Hierarchical Clustering page</b>
    </td>
  </tr>
</table>

#### Data Preprocessing
To ensure any dataset was ready for unsupervised machine learning, I created a function to preprocess the data.
- First, rows with missing data are excluded. There are different ways to handle missing data, but deletion was chosen here since the app can be used with any dataset and it would be difficult to make specific decisions.
- Categorical variables are encoded so they can be interpreted as numerical.
- If the data is labeled, the dataset is split between features and target variable.
- The data is centered so that each feature has a mean of zero.
- Then, each feature is scaled by its standard deviation.
These steps ensure that each feature makes an equal contribution to the model. Otherwise, features with naturally larger ranges would disproportionally influence the results.
```
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
```

## Instructions for Use
The app has been deployed and can be accessed with the URL below. To use the app, simply visit the following link:
<p align="center">
<a href="https://unsupervised-ml-explorer.streamlit.app"><img alt="Unsupervised ML Explorer App"
src="https://img.shields.io/badge/Unsupervised_ML_Explorer_App-73a3bd?style=for-the-badge"/></a> &nbsp;
</p>

If you want to download and run the code yourself, follow the instructions below:
1. Install Anaconda.
2. Clone this repository and cd into it.
```
git clone https://github.com/naraujodc/Araujo_Data_Science_Portfolio

cd MLUnsupervisedApp
```
3. In a new environment, install the dependencies.
```
conda activate new_env

pip install -r requirements.txt
```
4. In the terminal, make sure you are in `MLUnsupervisedApp`. If not, cd into it.
5. In the terminal, run the app on Streamlit.
```
streamlit run Unsupervised_ML.py
```

## Files
- `Unsupervised_ML.py` &rarr; This Python file contains the code for the app's landing page.
- `Supporting_Functions.py` &rarr; This Python file contains all functions I defined for the app and used across the pages.
- `pages` &rarr; This folder contains the three pages in the app, one for each unsupervised ML model.
  - `1_Principal_Component_Analysis.py` &rarr; This Python file contains the code for the PCA page.
  - `2_K-Means_Clustering.py` &rarr; This Python file contains the code for the K-Means Clustering page.
  - `3_Hierarchical_Clustering.py` &rarr; This Python file contains the code for the Hierarchical Clustering page.
- `requirements.txt` &rarr; List of required Python libraries.
- `README.md` &rarr; Project documentation.
- `images` &rarr; This directory stores the images used in the documentation.
- `.streamlit` &rarr; This directory stores `config.toml`, a file containing the custom theme I made for the app.

## Datasets Description
The demo datasets provided in the app are [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) from `scikit-learn`.
- **Breast cancer wisconsin (diagnostic) dataset:** Used to predict whether a tumor is malignant or benign according to 30 (numerical) predictive variables.
- **Iris plants dataset:** Used to predict whether an Iris plant is from the species Setosa, Versicolour, or Virginica according to 4 (numerical) predictive variables.
-  **Wine recognition dataset:** Used to predict whether a wine was manufactured by cultivator 0, 1, or 2 in Italy using 13 (numerical) predictive variables.

## References
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Stack Overflow](https://stackoverflow.com/questions)
