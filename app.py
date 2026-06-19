import streamlit as st
import pandas as pd

st.title("ML Playground")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write("Rows and Columns:", df.shape)
    st.write("Column Data Types:")
    st.write("Datatypes:",df.dtypes)
