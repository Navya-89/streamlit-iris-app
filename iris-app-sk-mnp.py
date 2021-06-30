import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

def      user_input():
  sepal_length = st.sidebar.slider('Sepal Length',4.3,7.9,5.4)
  sepal_width = st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
  petal_length = st.sidebar.slider('Petal Length',1.0,6.9,4.6)
  petal_width = st.sidebar.slider('Sepal Length',0.1,2.5,1.5)  # min,max,initial
  data = {'sepal_length': sepal_length,
          'sepal_width':sepal_width,
          'petal_length': petal_length,
          'petal_width': petal_width}
  features = pd.DataFrame(data,index=[0])
  return features        

st.write('# Simple **Iris Flower** prediction app')
st.write("# Iris Dataset")
st.write("Number of classses : 3")
st.write("Classifier: KNN")
st.sidebar.header("User input parameters")
st.subheader("User Input parameters")

df = user_input()
st.write(df)

iris = datasets.load_iris()
x = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)
y_pred = model.predict(df)

st.subheader("Class Labeles and their target specifiers")
st.write(iris.target_names)

st.subheader("Prediction")
st.write(iris.target_names[y_pred])