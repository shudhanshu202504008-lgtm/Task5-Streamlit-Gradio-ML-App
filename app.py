import streamlit as st
import pickle
import numpy as np

# Model loading with pickle
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Task 5 - Streamlit Interactive ML App")
st.write("Deploy your trained model with sliders")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 3.8)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(features)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted: {species[pred[0]]}")
