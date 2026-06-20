import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.title("ML Playground")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write("Rows and Columns:", df.shape)
    st.write("Column Data Types:")
    st.write("Datatypes:",df.dtypes)
    target = st.selectbox("Select a column to analyze", df.columns, key="column_selector")
    y = df[target]
    X = df.drop(target, axis=1)
    st.write(f"Analyzing column: {target}")
    st.write(f"Features: {X.shape[1]} columns, Target: {target}")
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    missing_value_strategy = st.selectbox("How to handle missing values?", ["Drop rows", "Fill with mean"], key="missing_value_strategy")
    if missing_value_strategy == "Drop rows":
        X = X.dropna()
    else:
        X = X.fillna({col: X[col].mean() if X[col].dtype in ['float64', 'int64'] else X[col].mode()[0] for col in X.columns})
    X = pd.get_dummies(X)
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_size_slider")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
