import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Datos de entrenamiento
entrada_entrenamiento = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
salida_entrenamiento = np.array([0,1,1,3,1,3,5,7,1,3,5,7,9,11,13,15,1,3,5,7,9,11,13,15,17,19,21,23,25])

# Ajustar la longitud de salida_entrenamiento
salida_entrenamiento = salida_entrenamiento[:len(entrada_entrenamiento)]

# Construir el modelo
modelo = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(1, activation='linear')
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
modelo.fit(entrada_entrenamiento, salida_entrenamiento, epochs=5000, verbose=0)

# Función para predecir la posición de Josephus
def predecir_posicion(numero):
    entrada = np.array([numero])
    prediccion = modelo.predict(entrada)
    return int(round(prediccion[0][0]))

# Ejemplos de predicciones
for num in [1, 2, 6, 10, 20]:
    posicion_predicha = predecir_posicion(num)
    print(f"Para el número {num}, comenzar en la posición {posicion_predicha}")

# Ejemplo de entrada arbitraria
numero_arbitrario = 6
posicion_arbitraria = predecir_posicion(numero_arbitrario)
print(f"Para el número {numero_arbitrario}, comenzar en la posición {posicion_arbitraria}")
