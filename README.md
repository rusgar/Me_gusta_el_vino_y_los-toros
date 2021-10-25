# White-or-red-wine-app

![White-or-red-wine](https://github.com/rusgar/White-or-red-wine/blob/main/image/red-and-wine.jpg)


# Project Overview

Wine is an alcoholic beverage made from fermented grapes. Yeast consumes the sugar in the grapes and converts it to ethanol, carbon dioxide, and heat. It is a pleasant tasting alcoholic beverage, loved cellebrated . It will definitely be interesting to analyze the physicochemical attributes of wine and understand their relationships and significance with wine quality and types classifications. To do this, We will proceed according to the standard Machine Learning and data workflow models like the TPOT API model and devlop an app 

# Data Overview

Dataset is from Kaggle. This datasets is related to red variants of the Portuguese "Vinho Verde" wine.Vinho verde is a unique product from the Minho (northwest) region of Portugal. Medium in alcohol, is it particularly appreciated due to its freshness (specially in the summer). The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones).

[Kaggle the White-Wine](https://www.kaggle.com/piyushagni5/white-wine-quality)

[Kaggle the Red-Wine](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

After cleaning the two datasets, using the appropriate tools, we join them and check with the two models the differences between clean or unclean data

# Dependences
Here we can find the libraries we will use in order to develop a solution for this problem.

**numpy|pandas**: Will help us treat the data. 

* [pandas](http://pandas.pydata.org/)
* [numpy](http://www.numpy.org/)

**matplotlib|seaborn**: Will help us plot the information so we can visualize it in different ways and have a better understanding of it.

* [matplotlib](http://matplotlib.org/)
* [seaborn](https://stanford.edu/~mwaskom/software/seaborn/)

**the interquartile range (IQR)**: We will use it to disperse the data and eliminate outliers and out-of-range values.

* [IQR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.iqr.html)

**sklearn**: Will provide all necessary tools to train our models and test them afterwards.

* [sklearn](http://scikit-learn.org/stable/)

**math**: Will provide some functions we might want to use when testing our models (sqrt)
* [math](http://www.w3big.com/es/python/func-number-sqrt.html)

**streamlit** : It is the library that makes it easy to create web applications to display results of your data analysis.
* [streamlit](https://streamlit.io/)


##### Attributes Information


Input variables (based on physicochemical tests):
1. fixed acidity 
2. volatile acidity 
3. citric acid 
4. residual sugar 
5. chlorides 
6. free sulfur dioxide 
7. total sulfur dioxide 
8. density 
9. pH 
10. sulphates 
11 .alcohol 
Output variable (based on sensory data): 
12. quality (score between 3 to 9)

![Calidad del Vino](https://github.com/rusgar/White-or-red-wine/blob/main/image/calidad-vino.png)

# The Data Science

The data science workflow is a non-linear and iterative task which requires many skills and tools to cover the whole process. From framing your business problem to generating actionable insights. It includes following steps
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Exploratory Data Analysis
5. Data Modeling
6. Model Evaluation
7. Model Deployment

[Documentacion](https://www.educba.com/data-science-lifecycle/)

![data-science](https://github.com/rusgar/White-or-red-wine/blob/main/image/data-science.png)


Here I am going to do a simple machine learning with the help of Streamlit to predict the quality of the wine (as in the dataset).
How to use machine learning to determine what physicochemical properties can make a wine "good", "medium" or "bad" and develop a web application with the help of streamlit to predict the quality of the wine?

### Understanding, preparing and exploring data analysis
Once the data is loaded, cleanliness and structure, analysis, type of quality, its graphs, we reduce the atypical data thanks to the (IQR), we observe the correlation of its characteristics, we save it in another .csv

![Correlacion del Vino Tinto](https://github.com/rusgar/White-or-red-wine/blob/main/image/Corr-red.png)

![Correlacion lineal entre densidad y el acido fijo](https://github.com/rusgar/White-or-red-wine/blob/main/image/corr-lineal.png)

### Data Modeling and evaluation

Union of the two datasets of white and red wine, elimination of duplicates, check the correlation regarding the quality, testing and training of the model, using the decision tree algorithms, with a conclusion of an accuracy of 1.0, in turn, We decompose the graph to distinguish where each wine is and two models so that they tell us the results of the dataset and know the quality of the wine: low, medium, good.

![Grafica del color](https://github.com/rusgar/White-or-red-wine/blob/main/image/grafica-color.png)
![Grafica del color](https://github.com/rusgar/White-or-red-wine/blob/main/image/comp-corr.png)
![Grafica del color](https://github.com/rusgar/White-or-red-wine/blob/main/image/model-corr.png)
![Grafica del color](https://github.com/rusgar/White-or-red-wine/blob/main/image/pred-resul.png)

As we continue to investigate, we do two testing and training models of the original data and those of the result model

[Pairing](https://github.com/rusgar/White-or-red-wine/blob/main/metrics.txt)

![Pairing](https://github.com/rusgar/White-or-red-wine/blob/main/image/pairing.png)

![Score without the IQR](https://github.com/rusgar/White-or-red-wine/blob/main/image/accuracy-0.86.png)

![Score with the IQR](https://github.com/rusgar/White-or-red-wine/blob/main/image/accuracy-1.png)




### Model Deployment
We are going to use **streamlit**, but before our development model, we use to be more precise **TPOT**, **joblib** to save the model and use it and with **AutoML** (automatic learning) we configure our final app

![AutoML](https://github.com/rusgar/White-or-red-wine/blob/main/image/AutoML.jpg)

[Streamlit](https://streamlit.io/)

[TPOT](http://epistasislab.github.io/tpot/)

[Joblib](https://scikit-learn.org/stable/modules/model_persistence.html)

[AutoML](https://www.automl.org/)


# Required Files
### Setup.sh
files with the extension. sh are operating system scripts. We can execute them from the command line or from the interface of our distribution, in this case we will introduce it in our root file

```
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Procfile
We place in our root project the **procfile** file is a mechanism to declare which commands your application executes on the platform

```
web: sh setup.sh && streamlit run app.py
```

### Requirements.txt

1. Open a command terminal
2. Navigate to the root of the project where you want to list the dependencies
3. Execute

```
pip freeze > requirements.txt
```

4. Open the file requirements.txt created, it will list all the package dependencies as well as the version of the package that your project requires to work:

To install this list of dependencies in any other Python installation you can run

```
pip install -r requirements.txt
```


## App Deployment
1. Open a command terminal
2. Navigate to the project folder where we have the app.py
3. Execute


```
mkdir -p ~/streamlit run app.py
```

A local host will be opened with the streamlit app running

# Production
[Streamlit](https://streamlit.io/cloud)

From this link we can upload our app in production without any difficulty, it will ask us to log in, and following the steps,

1- New App

2- Deploy an app

       2.1 Repository
  
       2.2 Branch
  
       2.3 Main file path
  
3- Deploy

4- After a few seconds on our right we will see how the commands are being executed ... to have our app with its url

# APP

![Python 3.9.2](https://img.shields.io/badge/Python_v3.9.2-F7D448?&logo=python&logoColor=356c9b)  ![Streamlit 1.1.0](https://img.shields.io/badge/-Streamlit_v1.1.0-orange?logo=streamlit&logoColor=#8A4A4F)  ![Jupyter 6.3.0](https://img.shields.io/badge/Jupyter_v6.3.0-black?&logo=Jupyter&logoColor=eb5f05) ![NUmpy](https://img.shields.io/badge/Numpy_v1.19.5-black?&logo=numpy&logoColor=4B73C9) ![Pandas](https://img.shields.io/badge/Pandas_v1.2.4-F7D448?&logo=pandas&logoColor=120751)
![Scikit learn](https://img.shields.io/badge/Scikit_learn_v0.24.2-3294C7?&logo=scikit-learn&logoColor=F09437)



![Matplotlib 3.4.3](https://img.shields.io/badge/Matplotlib-_v3.4.3-F7D448) ![TPOT 0.11.7](https://img.shields.io/badge/TPOT-_v0.11.7-0A2C74) ![Joblib 1.0.1](https://img.shields.io/badge/Joblib-_v1.0.1-DA5316) ![Pillow](https://img.shields.io/badge/Pillow-_v8.4.0-87DEF7)


![ML](https://img.shields.io/badge/Machine-Learning-blue) ![Status](https://img.shields.io/badge/Status-Completed-success)

[White-or-red-wine](https://share.streamlit.io/rusgar/white-or-red-wine/main/app.py)





