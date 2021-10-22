import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image



# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df):
    """  funcion preprocesa la entrada del usuario
        return type: pandas dataframe
    """
    df.wine_type = df.wine_type.map({'white':0, 'red':1})
    return df

def visualize_confidence_level(prediction_proba):
    """
    usa matplotlib para crear un gráfico de barras de inferencia renderizado con streamlit en tiempo real
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    nivel = pd.DataFrame(data = data,columns = ['Porcentaje'],index = ['Bajo','Mediano','Bueno'])
    
    ax = nivel.plot(kind='barh', figsize=(10, 5), color='#FB6942', zorder=10, width=0.5)
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

    ax.set_xlabel(" Nivel (%) Porcentaje", labelpad=3, weight='bold', size=12)
    ax.set_ylabel("Calidad del vino", labelpad=10, weight='bold', size=12)
    ax.set_title('Nivel de prediccion ', fontdict=None, loc='center', pad=None, weight='bold')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    return

st.write("""
# App de Preciccion de las clases de vino 
Esta aplicación predice la ** Calidad del vino ** mediante la entrada de ** características del vino ** a través del ** panel lateral **
""")

#Lectura de una imagen prediseñada
image = Image.open('../image/blanco-vs-tinto.png')
st.image(image, caption='Tinto o Blanco',use_column_width=True)

#colección de parámetros de entrada del usuario con barra lateral iluminada

st.sidebar.header('Introducir sus caracteristicas')


def get_user_input():
    """
    obtener la entrada del usuario usando el control deslizante de la barra lateral y el cuadro de selección
    return type : pandas dataframe

    """
    wine_type = st.sidebar.selectbox("Seleccione el tipo de vino",("white", "red"))
    fixed_acidity = st.sidebar.slider('fixed acidity', 3.8, 15.9, 7.21)
    volatile_acidity = st.sidebar.slider('volatile acidity', 0.08, 1.58, 0.33)
    citric_acid  = st.sidebar.slider('citric acid', 0.0, 1.66, 0.31)
    residual_sugar  = st.sidebar.slider('residual_sugar', 0.60, 65.8, 5.44)
    chlorides  = st.sidebar.slider('chlorides', 0.009, 0.611, 0.056)
    free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1, 289, 30.52)
    total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide', 6, 440, 115)
    density = st.sidebar.slider('density', 0.987, 1.038, 0.994)
    pH = st.sidebar.slider('pH', 2.72, 4.01, 3.21)
    sulphates = st.sidebar.slider('sulphates', 0.22, 2.0, 0.53)
    alcohol = st.sidebar.slider('alcohol', 8.0, 14.9, 10.49)
    
    features = {'wine_type': wine_type,
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