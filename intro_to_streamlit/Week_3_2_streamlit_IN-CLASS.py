# Import the Streamlit library
import streamlit as st

# Display a simple text message
st.title("Hello, streamlit!")
st.markdown("### This is my first streamlit app!")
# Display a large title on the app

# ------------------------
# INTERACTIVE BUTTON
# ------------------------

# Create a button that users can click.
# If the button is clicked, the message changes.
if st.button("Click me!"):
    st.write("You clicked the button. Nice work!")
else:
    st.write("Go ahead... click the button. I dare you.")

# ------------------------
# COLOR PICKER WIDGET
# ------------------------
color = st.color_picker("What's your favorite color?")
# Creates an interactive color picker where users can choose a color.
# The selected color is stored in the variable 'color'.

# Display the chosen color value
st.write(f"Your favorite color is {color}")
# ------------------------
# ADDING DATA TO STREAMLIT
# ------------------------

# Import pandas for handling tabular data
import pandas as pd

# Display a section title
st.title("Data")
# Create a simple Pandas DataFrame with sample data


# Display a descriptive message

# Display the dataframe in an interactive table.
# Users can scroll and sort the data within the table.

# ------------------------
# INTERACTIVE DATA FILTERING
# ------------------------

# Create a dropdown (selectbox) for filtering the DataFrame by city.
# The user selects a city from the unique values in the "City" column.

# Create a filtered DataFrame that only includes rows matching the selected city.

# Display the filtered results with an appropriate heading.
  # Show the filtered table

# ------------------------
# NEXT STEPS & CHALLENGE
# ------------------------
st.title("Feedback")
selected = st.checkbox("Rate this webpage")
if selected == True:
    st.slider("Rate this webpage from 0 to 10", min_value=0, max_value=10)

# Play around with more Streamlit widgets or elements by checking the documentation:
# https://docs.streamlit.io/develop/api-reference
# Use the cheat sheet for quick reference:
# https://cheat-sheet.streamlit.app/

### Challenge:
# 1️⃣ Modify the dataframe (add new columns or different data).
# 2️⃣ Add an input box for users to type names and filter results.
# 3️⃣ Make a simple chart using st.bar_chart().