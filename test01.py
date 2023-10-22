# Importa las bibliotecas necesarias
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Genera datos aleatorios para X
np.random.seed(0)
X = np.random.rand(100, 1)

# Calcula Y basado en una relación lineal con un poco de ruido
Y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Crea un modelo de regresión lineal
model = LinearRegression()

# Ajusta el modelo a los datos
model.fit(X, Y)

# Obtiene los coeficientes de la ecuación de regresión
slope = model.coef_[0][0]
intercept = model.intercept_[0]

# Imprime la ecuación de regresión
print("Ecuación de regresión lineal: Y = {:.2f}X + {:.2f}".format(slope, intercept))

# Grafica los datos y la línea de regresión
plt.scatter(X, Y, label='Datos')
plt.plot(X, model.predict(X), color='red', label='Regresión Lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()