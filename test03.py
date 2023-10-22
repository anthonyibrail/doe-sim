import numpy as np
import pandas as pd

# Definimos los factores y sus niveles
factores = ["A", "B", "C", "D", "E", "F", "G"]
niveles = ["-1", "1"]

# Generamos las combinaciones de tratamientos
combinaciones = np.array(np.meshgrid(*[niveles for _ in range(len(factores))], indexing="ij")).T.reshape(-1, len(factores))

# Definimos las variables de respuesta
variables_respuesta = np.random.uniform(0, 1.5, size=combinaciones.shape[0])

# Reducción de la variable de respuesta a solo 4 decimales
variables_respuesta = np.around(variables_respuesta, decimals=4)

# Generamos los datos del experimento
datos = np.column_stack((combinaciones, variables_respuesta))

# Guardamos los datos en un archivo de Excel
df = pd.DataFrame(datos, columns=factores + ["Respuesta"])
df.to_excel("datos.xlsx")
