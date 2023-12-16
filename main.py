import streamlit as st
import pickle
import numpy as np
#import sklearn

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
scaler = data["scaler"]

st.title("Would you survive Titanic disaster?")
st.write("""### We need some information to predict the output""")

genders = {"male", "female"}

embarked_ports = {"Chernourg", "Queenstown", "Southampton"}

gender = st.selectbox("Gender", genders)
embarked = st.selectbox("Embarked", embarked_ports)
pclass = st.slider("Ticket class", 1, 3, 1)
age = st.number_input("Age", 1, 100, 3)
sibsp = st.number_input("Siblings and spouses", 0, 10, 0)
parch = st.number_input("Parents and children", 0, 10, 0)
fare = st.slider("Fare", 1, 200, 20)

ok = st.button("Calculate prediction")
if ok:
    X = np.array([[pclass, 1 if gender=="female" else 0, age, sibsp, parch, fare, 1 if embarked=="Chernourg" else 0, 1 if embarked=="Queenstown" else 0, 1 if embarked=="Southampton" else 0]])
    X = scaler.transform(X)

    prediction = model.predict(X)
    if prediction==1:
        st.subheader("You probably would survive the disaster!")
    else:
        st.subheader("You wouldn't survive!")
