import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("ML Playground")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write("Rows and Columns:", df.shape)
    #st.write("Column Data Types:")
    #st.write("Datatypes:",df.dtypes)
    #target = st.sidebar.selectbox("Select a column to analyze", df.columns, key="column_selector")
    
    cols_to_exclude = []
    for col in df.columns:
      if df[col].nunique() == len(df):
        cols_to_exclude.append(col)
    cols_to_exclude = st.sidebar.multiselect("Columns to exclude", df.columns, default=cols_to_exclude)   
    df = df.drop(columns=cols_to_exclude)

    target = st.sidebar.selectbox("Select a column to analyze", df.columns, key="column_selector")
    st.write(f"Analyzing column: {target}")
    #st.write("Missing Values:")
    #st.write(df.isnull().sum())
    missing_value_strategy = st.sidebar.selectbox("How to handle missing values?", ["Drop rows", "Fill with mean"], key="missing_value_strategy")
    if missing_value_strategy == "Drop rows":
       df_clean = df.dropna()
    else:
        df_clean = df.fillna({col: df[col].mean() if df[col].dtype in ['float64', 'int64'] else df[col].mode()[0] for col in df.columns})
    
    X = df_clean.drop(target, axis=1)
    y = df_clean[target]
    st.write(f"Features: {X.shape[1]} columns, Target: {target}")
    X = pd.get_dummies(X)
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_size_slider")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    st.sidebar.selectbox("Select a model to train", ["Logistic Regression","Decision Tree","KNN", "Random Forest"], key="model_selector")
    if st.sidebar.button("Train Model", key="train_button"):
        model_name = st.session_state.model_selector
        
        if model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        elif model_name == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
        elif model_name == "KNN":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
        elif model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Model trained successfully!")
        st.metric("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(pd.DataFrame(cm))
