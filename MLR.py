import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Funckje import percent_missing, regresive_model_split, run_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error,mean_absolute_error


st.title("Welcome to Machine learning")

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
        st.dataframe(split_df)

    fig, ax = plt.subplots(figsize=(10,4),dpi = 200)

    ax.scatter(
        split_df.index,
        split_df[y_col],
        c=split_df["split"].map({"Train": "blue", "Test": "red"})
    )

    ax.set_xlabel("Sorted compounds")
    ax.set_ylabel(y_col)

    st.pyplot(fig)

    if "split" in split_df.columns:

        train_df = split_df[split_df["split"] == "Train"]
        test_df = split_df[split_df["split"] == "Test"]

        X_train = train_df.drop(columns=[y_col, "split"])
        y_train = train_df[y_col]

        X_test = test_df.drop(columns=[y_col, "split"])
        y_test = test_df[y_col]

        st.subheader("Model Selection")

        model_name = st.selectbox(
            "Choose model",
            ["MLR", "PCR", "PLS", "SVR", "RFR"]
        )

        if model_name == "MLR":
            model = LinearRegression()

        elif model_name == "PCR":

            n_comp = st.slider(
                "Number of PCA components",
                1,
                min(X_train.shape[1], 20),
                5
            )

            model = Pipeline([
                ("pca", PCA(n_components=n_comp)),
                ("regressor", LinearRegression())
            ])

        elif model_name == "PLS":

            n_comp = st.slider(
                "Number of PLS components",
                1,
                min(X_train.shape[1], 20),
                5
            )

            model = PLSRegression(n_components=n_comp)

        elif model_name == "SVR":

            C = st.number_input("C", value=1.0)

            model = SVR(
                kernel="rbf",
                C=C
            )

        elif model_name == "RFR":

            n_estimators = st.slider(
                "Number of trees",
                50,
                1000,
                200
            )

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42
            )

        if st.button("Run model"):
            model, results, y_test_pred, y_train_pred = run_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test)

        col1, col2 = st.columns([1, 2])

        with col1:

            metrics_df = pd.DataFrame(
                {
                    "Metric": results.keys(),
                    "Value": results.values()
                }
            )

            st.subheader("Model Performance")
            st.dataframe(
                metrics_df,
                hide_index=True,
                use_container_width=True
            )

        with col2:

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.scatter(
                y_train, y_train_pred,
                label="Train",
                alpha=0.6
            )

            ax.scatter(
                y_test, y_test_pred,
                label="Test",
                alpha=0.6
            )

            # wspólna oś odniesienia
            all_vals = np.concatenate([
                y_train.values,
                y_test.values,
                y_train_pred,
                y_test_pred
            ])

            min_val = all_vals.min()
            max_val = all_vals.max()

            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                '--',
                color='black')

            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.set_title("Observed vs Predicted")

            ax.legend()
            ax.grid(True)

            st.pyplot(fig)