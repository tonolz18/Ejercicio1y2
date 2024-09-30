# Importar las librerías necesarias
from sklearn.datasets import fetch_openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Descargar la base de datos Titanic
titanic = fetch_openml(name='titanic', version=1, as_frame=True)

# Descripción básica de las variables
print(titanic['data'].info())

# Descripción de las variables numéricas
print(titanic['data'].describe())

# Gráfico de barras para la variable 'sex'
sns.countplot(x='sex', data=titanic['data'])
plt.title('Distribución de Hombres y Mujeres')
plt.show()

# Histograma para la variable 'age'
sns.histplot(titanic['data']['age'].dropna(), kde=True)
plt.title('Distribución de la Edad de los Pasajeros')
plt.show()

# Gráfico de caja para la variable 'fare'
sns.boxplot(x=titanic['data']['fare'])
plt.title('Distribución de las Tarifas')
plt.show()

# Gráfico de dispersión entre 'age' y 'fare'
sns.scatterplot(x='age', y='fare', data=titanic['data'])
plt.title('Relación entre Edad y Tarifa')
plt.show()

# Obtener los parámetros estadísticos para todas las variables
print(titanic['data'].describe(include='all'))
