import streamlit as st
import pandas as pd

# ======================
# setting up the web page
# ======================

# set page configurations
st.set_page_config(
    page_title = "Netflix Movies and TV Shows",
    page_icon = ":popcorn:",
    layout = "wide"
)

# display web page title
st.title("Netflix Movies and TV Shows")

# create button linking to original dataset
st.link_button("View dataset on Kaggle",
               "https://www.kaggle.com/datasets/anandshaw2001/netflix-movies-and-tv-shows?resource=download")

# write a brief description
st.markdown(
    """
    This dataset lists all movies and TV shows available on Netflix.
    Here, you can navigate through the data and filter it by different attributes
    such as media type, categories, and rating.
    """
)

# ======================
# setting up the dataframe
# ======================

# add dataset as a pandas dataframe
df = pd.read_csv("data/netflix_titles.csv")

# rename columns
df = df.rename(columns = {
    "type" : "Type",
    "title" : "Title",
    "director" : "Director",
    "cast" : "Cast",
    "country" : "Country",
    "date_added" : "Date Added",
    "release_year" : "Release Year",
    "rating" : "Rating",
    "duration" : "Duration",
    "listed_in" : "Categories",
    "description" : "Description"
})

# set show title as the index
df.set_index("Title", inplace=True)

# delete show_id column
df = df.drop("show_id", axis=1)

# ======================
# creating filters for the dataframe
# ======================

# setting the filtered df before filters
filtered_df = df

# create columns to place the widgets
col1, col2, col3 = st.columns([0.2, 0.55, 0.25])

## type
# creating a segmented control widget for type
media_types = df["Type"].unique()

with col1:
    media_type = st.segmented_control(
        "Filter by media type:",
        options = media_types,
        selection_mode = "multi",
        default = ["Movie", "TV Show"]
    )

# filter dataset according to chosen type
if media_type:
    filtered_df = df[df["Type"].isin(media_type)]

## categories
# creating a multiselect button for categories
categories = df["Categories"].unique()

with col2:
    category = st.multiselect(
        "Filter by categories:",
        options = categories
    )

# filter dataset according to categories
if category:
    filtered_df = df[df["Categories"].isin(category)]

## rating
# creating a multiselect button for ratings
ratings = df["Rating"].unique()

with col3:
    rating = st.multiselect(
        "Filter by rating:",
        options = ratings
    )

# filter dataset according to ratings
if rating:
    filtered_df = df[df["Rating"].isin(rating)]

# display dataframe
st.dataframe(filtered_df)

# link to my github
st.markdown("Developed by [Natália Araújo](https://github.com/naraujodc)")