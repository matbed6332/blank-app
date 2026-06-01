import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Funckje import percent_missing

st.title("Welcome to MLR modeling")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:

    # Read excel
    df = pd.read_excel(uploaded_file, sheet_name=0)

    st.write("### Original Data")
    st.dataframe(df)
    st.divider()
    
    y_col = st.selectbox("Select target column (y)", options=df.columns)

    index_col = st.selectbox("Select index column (optional)", options=["None"] + list(df.columns))

    drop_cols = st.multiselect("Select columns to drop", options=df.columns)
    
    #Barplot
    percent_nan = percent_missing(df)
    if len(percent_nan) > 0:

        st.write("### Missing Values (%)")

        fig, ax = plt.subplots(figsize=(16, 8), dpi=200)

        sns.barplot(
            x=percent_nan.index,
            y=percent_nan.values,
            palette="inferno",
            ax=ax
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylim(0, 100)

        st.pyplot(fig)

    else:
        st.success("No missing values detected.")

    processed_df = df.copy()

    # Set index
    if index_col != "None":
        processed_df = processed_df.set_index(index_col)

    # Drop columns
    processed_df = processed_df.drop(columns=drop_cols, errors="ignore")

    st.write("### Processed Data")
    st.dataframe(processed_df)

    test_every = st.number_input(
    "Take every Nth compound to test set",
    min_value=2,
    value=3,
    step=1)
    
    if y_col:
        # sortowanie po Y
        sorted_df = processed_df.sort_values(by=y_col).reset_index(drop=True)

        # podział
        split_df = regresive_model_split(sorted_df.copy(), test_every)

        st.write(split_df["split"].value_counts())
        st.dataframe(split_df.head())