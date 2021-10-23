import numpy as np 
import pandas as pd 
import joblib 
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("src/modelo_wine.joblib","rb"))

def data_preprocessor(df):
    """función preprocesa la entrada del usuario
        return type: pandas dataframe
    """
    df.color = df.color.map({'white':0, 'red':1})
    return df

def visualize_confidence_level(prediction_proba):
    """
    crear un gráfico de barras de inferencia renderizado con streamlit en tiempo real 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Bajo','Mediano','Bueno'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#FB6942', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    value = ax.get_xticks()
    for tick in value:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.8, color='#FB6942', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Wine Quality", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.pyplot()
    return

st.write("""
Esta aplicación predice la ** Calidad del vino ** mediante la entrada de ** características del vino ** a través del ** panel lateral ** 
""")

#leer en la imagen del vino y renderizar con streamlit
image = Image.open('blanco-vs-tinto.png')
st.image(image, caption='Tinto o Blanco',use_column_width=True)

st.sidebar.header('Introduzca sus cualidades') #colección de parámetros de entrada del usuario con side bar


def get_user_input():
    """
    obtener la entrada del usuario usando sidebar slider and selectbox 
    return type : pandas dataframe

    """
    color = st.sidebar.selectbox("Seleccione el tipo de Vino",("white", "red"))
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.8, 12.3, 7.12)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.08, 1.05, 0.32)
    citric_acid  = st.sidebar.slider('Citric Acid', 0.000, 0.730, 0.305)
    residual_sugar  = st.sidebar.slider('Residual Sugar', 0.6, 22.0, 5.4)
    chlorides  = st.sidebar.slider('Chlorides', 0.015, 0.119, 0.051)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1.00, 80.00, 30.12)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6.00, 255.00, 115.17)
    density = st.sidebar.slider('Density', 0.987, 1.001, 0.994)
    pH = st.sidebar.slider('Ph', 2.82, 3.68, 3.21)
    sulphates = st.sidebar.slider('Sulphates', 0.22, 0.98, 0.51)
    alcohol = st.sidebar.slider('Alcohol', 8.4, 14.2, 10.49)
    
    features = {'color': color,
            'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
            }
    data = pd.DataFrame(features,index=[0])

    return data

input_df = get_user_input()
processed_input = data_preprocessor(input_df)

st.subheader('Caracteristicas Introducidas')
st.write(input_df)

predict = model.predict(processed_input)
predict_proba = model.predict_proba(processed_input)

visualize_confidence_level(predict_proba)