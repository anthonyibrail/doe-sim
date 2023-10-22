import numpy as np
import pandas as pd
import statsmodels.api as sm

# Establecer la semilla para reproducibilidad
np.random.seed(0)

# Definir los niveles de los factores
n = 100  # Número de observaciones
levels = 2  # Número de niveles para cada factor (puedes ajustarlo según tus necesidades)

# Generar datos aleatorios para los factores A, B, C, D, E, F, G
A = np.random.randint(0, levels, size=n)
B = np.random.randint(0, levels, size=n)
C = np.random.randint(0, levels, size=n)
D = np.random.randint(0, levels, size=n)
E = np.random.randint(0, levels, size=n)
F = np.random.randint(0, levels, size=n)
G = np.random.randint(0, levels, size=n)

# Generar datos aleatorios para la respuesta (variable dependiente)
# Aquí puedes definir tu propia relación en función de los factores y las interacciones relevantes
# En este ejemplo, se usa una relación simple.
y = 2 * A + 3 * C + 4 * D + 5 * G + 1.5 * A * C * D * G + np.random.normal(0, 1, size=n)

# Crear un DataFrame con los datos
data = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'Response': y})

# Ajustar el modelo de regresión lineal
formula = 'Response ~ A + C + D + G + A:C:D:G'
model = sm.OLS.from_formula(formula, data=data).fit()

# Mostrar la ecuación de los factores y las interacciones relevantes
print(model.summary())