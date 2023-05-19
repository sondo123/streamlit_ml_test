import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

st.title("Streamlit Example")

st.write("""
# Explore KNN for different datasets
Which K works best?
""")

dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

st.write(f"## {dataset_name} Dataset")

data = None
if dataset_name == "Wine":
    data = datasets.load_wine()
elif dataset_name == "Iris":
    data = datasets.load_iris()
else:
    data = datasets.load_breast_cancer()

X = data.data
Y = data.target

st.write(f"Shape of dataset: {X.shape}")
st.write(f"Number of classes: {len(np.unique(Y))}")

# create a DataFrame to be able to use pandas functions
df = pd.DataFrame(X, columns = data.feature_names)

K = st.sidebar.slider("K", 1, 15)

# Display data schema and sample row button
if st.sidebar.button('Display data schema and sample row'):
    st.write(df.info())
    st.write(df.head(1))

clf = KNeighborsClassifier(n_neighbors=K)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

st.write("Accuracy: ", accuracy_score(Y_test, Y_pred))

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file, header=None, sep=';')
    new_data_pred = clf.predict(new_data)
    pca = PCA(2)
    pca.fit(X_train)  # Fit the PCA on training data first
    new_data_projected = pca.transform(new_data)
else:
    new_data_projected = None

if st.button("Plot"):
    pca = PCA(2)
    pca.fit(X_train)
    X_projected = pca.transform(X_test)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=Y_pred, alpha=0.8, cmap="viridis")
    if new_data_projected is not None:
        new_x1 = new_data_projected[:, 0]
        new_x2 = new_data_projected[:, 1]
        plt.scatter(new_x1, new_x2, c=new_data_pred, alpha=1, cmap="viridis", marker='*')

    plt.colorbar()
    st.pyplot(fig)
