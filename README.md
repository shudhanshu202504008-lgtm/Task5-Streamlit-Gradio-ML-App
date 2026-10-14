import streamlit as st
import pickle
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Model loading with joblib/pickle - Task 5 feature
MODEL_FILE = 'iris_model.pkl'

# Agar pkl file nahi hai to usi time bana do (Mobile ka jugad)
if not os.path.exists(MODEL_FILE):
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
else:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

st.title("Task 5 - Streamlit / Gradio Interactive ML Web App")
st.write("Deploy your trained model into simple interactive web app with input sliders and real-time prediction display")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 3.8)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(features)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Flower: {species[pred[0]]}")
