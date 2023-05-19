import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Filling missing values with mean of their respective column
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col].fillna(df[col].mean(), inplace=True)


    # If there are categorical features, convert them to numerical
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    # Normalize all the features to be in range [0, 1]
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

    return df


def main():
    st.title("Machine Learning Platform")
    st.write("""
    Upload your data and select the problem type and algorithm to get started
    """)

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv", "xlsx"])
    problem_type = st.selectbox("Select problem type", ("Classification", "Regression", "Clustering"))

    if uploaded_file is not None:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)

        # Preprocess the data
        data = preprocess_data(data)
        
        # List of models
        classification_models = {"K-Nearest Neighbors": KNeighborsClassifier(), "Random Forest": RandomForestClassifier(), "Logistic Regression": LogisticRegression()}
        regression_models = {"Random Forest": RandomForestRegressor(), "Linear Regression": LinearRegression()}
        clustering_models = {"K-Means": KMeans(n_clusters=3)}  # A default of 3 clusters is used here

        algorithm_name = None
        if problem_type == "Classification":
            algorithm_name = st.selectbox("Select algorithm", list(classification_models.keys()))
        elif problem_type == "Regression":
            algorithm_name = st.selectbox("Select algorithm", list(regression_models.keys()))
        elif problem_type == "Clustering":
            algorithm_name = st.selectbox("Select algorithm", list(clustering_models.keys()))

        # Model selection
        model = None
        if algorithm_name is not None:
            if problem_type == "Classification":
                model = classification_models[algorithm_name]
            elif problem_type == "Regression":
                model = regression_models[algorithm_name]
            elif problem_type == "Clustering":
                model = clustering_models[algorithm_name]

        # Handling of labels
        if problem_type != "Clustering":
            if 'y' in data.columns:
                le = LabelEncoder()
                data['y'] = le.fit_transform(data['y'])
                y = data['y']
                X = data.drop('y', axis=1)
            else:
                st.write("No 'y' label found in data. Please ensure your label column is named 'y'.")
        else:  # No labels for clustering
            X = data

        # Train/test split and model training
        if st.button("Train Model"):
            if model is not None:
                if 'y' in data.columns:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # In case of no errors, display success message, otherwise display error message
                    try:
                        st.success("Model successfully trained!")
                        
                        # If problem type is regression, display plot button and metrics, add legends based on the labels
                        if problem_type == "Regression":
                            if st.button("Plot Regression"):
                                try:
                                    y_pred = model.predict(X_test)
                                    plt.scatter(X_test.iloc[:, 0], y_test, color='red', label='Actual')
                                    plt.scatter(X_test.iloc[:, 0], y_pred, color='blue', label='Predicted')
                                    plt.xlabel("X")
                                    plt.ylabel("Y")
                                    plt.title("Actual vs Predicted")
                                    plt.legend()
                                    st.pyplot()
                                except:
                                    st.error("Error in plotting. Please check your data.")
                                
                        # If problem type is classification, display confusion matrix and accuracy and plot button and scatter plot
                        if problem_type == "Classification":
                            if st.button("Show Confusion Matrix"):
                                try:
                                    y_pred = model.predict(X_test)
                                    from sklearn.metrics import confusion_matrix
                                    st.write(confusion_matrix(y_test, y_pred))
                                except:
                                    st.error("Error in plotting. Please check your data.")
                                
                    except:
                        st.error("Error in training model. Please check your data.")
                else:
                    st.write("No 'y' label found in data. Please ensure your label column is named 'y'.")
                    
        #  If problem type is clustering, display plot button and scatter plot
        if problem_type == "Clustering" and model is not None:
            if st.button("Train and Plot Clustering"):
                try:
                    model.fit(X)
                    y_pred = model.predict(X)
                    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='viridis')
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.title("Actual vs Predicted")
                    st.pyplot()
                except:
                    st.error("Error in plotting. Please check your data.")
if __name__ == "__main__":
    main()
