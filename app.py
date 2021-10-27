import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.linear_model import LogisticRegression


st.write("""
# White-or-red-wine
### Which one to choose at all times?
""")


# Load  model 
model = joblib.load(open("src/modelo_wine.joblib", "rb"))


def data_preprocesador(df):
    """función preprocesa la entrada del usuario
        return type: pandas dataframe
    """
    df.color = df.color.map({'white': 0, 'red': 1})
    return df


def visualizacion(prediction_proba):
    """
    crear un gráfico de barras de inferencia renderizado con streamlit en tiempo real 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data=data, columns=['Percentage'], index=[
                                   'Bajo', 'Mediano', 'Bueno', 'Excelente'])
    ax = grad_percentage.plot(kind='barh', figsize=(
        8, 6), color='#FB6942', zorder=30, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")

    value = ax.get_xticks()
    for tick in value:
        ax.axvline(x=tick, linestyle='dashed',
                   alpha=0.9, color='#FB6942', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level",
                  labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Wine Quality", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None,
                 loc='center', pad=None, weight='bold')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.pyplot()
    return


st.write("""Esta aplicación predice la ** Calidad del vino ** mediante la entrada de ** características del vino ** a través del ** panel lateral ** """)

# leer en la imagen del vino y renderizar con streamlit
image = Image.open('image/blanco-vs-tinto.png')
st.image(image, caption='Tinto o Blanco', use_column_width=True)

codigo = st.expander('¿Necesitas Ayuda? 👉')
with codigo:
    st.markdown(
        "Encontraras todas la informacion del dataset en [Rusgar](https://github.com/rusgar/White-or-red-wine), estamos para ayudar ")

# colección de parámetros de entrada del usuario con side_bar
st.sidebar.header('Introduzca sus cualidades')

dataset = st.selectbox('Haz tu eleccion', ('Conjunto', 'White', 'Red'))


def get_data(dataset):
    data_wine = pd.read_csv('data/df_wine.csv')
    data_white = pd.read_csv('data/wine_final_white_todo.csv')
    data_red = pd.read_csv('data/wine_final_red_todo.csv')
    if dataset == 'Conjunto':
        data = data_wine
    else:
        data = data_red
        if dataset == 'White':
            data = data_white
        else:
            data = data_red

    return data


data_heatmap = get_data(dataset)
data = get_data(dataset)


def get_dataset(dataset):
    bins = (1, 6, 10)
    groups = ['1', '2']
    data['quality'] = pd.cut(data['quality'], bins=bins, labels=groups)
    x = data.drop(columns=['quality'])
    y = data['quality']
    return x, y


x, y = get_dataset(data)
st.write('Conjunto de datos:', data.shape)
with st.expander('Visualizacion'):
    plot = st.selectbox('Selecione el tipo PLot',
                        ('Histogram', 'Box Plot', 'Heat Map'))

    if plot == 'Heat Map':
        fig1 = plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(data_heatmap.corr()[['quality']].sort_values(by='quality', ascending=False), vmin=-1,
                              vmax=1, annot=True)
        heatmap.set_title('Correlacion respecto a la calidad',
                          fontdict={'fontsize': 20}, pad=20)
        st.pyplot(fig1)
    else:
        feature = st.selectbox('Selecione su caracteristica', ('fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                                               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                                               'pH', 'sulphates', 'alcohol'))
        if plot == 'Histogram':
            fig2 = plt.figure(figsize=(7, 5))
            plt.xlabel(feature)
            sns.distplot(x[feature])
            st.pyplot(fig2)
        else:
            fig3 = plt.figure(figsize=(7, 5))
            plt.ylabel(feature)
            plt.boxplot(x=x[feature])
            st.pyplot(fig3)


def get_user_input():
    """
    obtener la entrada del usuario usando sidebar slider and selectbox 
    return type : pandas dataframe

    """
    color = st.sidebar.selectbox(
        "Seleccione el tipo de Vino", ("white", "red"))
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.8, 12.3, 7.12)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.08, 1.05, 0.32)
    citric_acid = st.sidebar.slider('Citric Acid', 0.000, 0.730, 0.305)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.6, 22.0, 5.4)
    chlorides = st.sidebar.slider('Chlorides', 0.015, 0.119, 0.051)
    free_sulfur_dioxide = st.sidebar.slider(
        'Free Sulfur Dioxide', 1.00, 80.00, 30.12)
    total_sulfur_dioxide = st.sidebar.slider(
        'Total Sulfur Dioxide', 6.00, 255.00, 115.17)
    density = st.sidebar.slider('Density', 0.967, 1.001, 0.994)
    pH = st.sidebar.slider('Ph', 2.82, 3.68, 3.21)
    sulphates = st.sidebar.slider('Sulphates', 0.22, 0.98, 0.51)
    alcohol = st.sidebar.slider('Alcohol', 8.4, 14.2, 10.49)

    nombres = {'color': color,
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
    data = pd.DataFrame(nombres, index=[0])

    return data


input_df = get_user_input()
procesar_input = data_preprocesador(input_df)


st.write(input_df)

data = pd.read_csv("data/df_wine.csv")
X = np.array(data[['color', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'alcohol', 'sulphates']])
Y = np.array(data['quality'])
model = LogisticRegression()
model.fit(X, Y)
st.subheader('Etiquetas de clase y su número de índice correspondiente')
st.write(pd.DataFrame({
    'wine quality': [4, 5, 6, 7]}))

predict = model.predict(input_df)
predict_probability = model.predict_proba(input_df)

st.subheader('Probabilidad que salga segun su calidad')
st.write(predict_probability)


predict = model.predict(procesar_input)
predict_probability = model.predict_proba(procesar_input)

visualizacion(predict_probability)

st.subheader('Prediccion del vino')

type_labels = {
    4: 'Malo',
    5: 'Mediano',
    6: 'Bueno',
    7: 'Excelente'
}
st.write(
    type_labels.get(model.predict(input_df)[0]))

st.write(predict)
st.markdown('4:Malo  5:Mediano  6:Bueno  7:Excelente')


"""
Created on Wed Oct 27 18:05:00 2022
@author: Rusgar
"""
