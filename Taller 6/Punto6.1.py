import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Generar valores aleatorios para x
np.random.seed(42)  # Para reproducibilidad
X = np.random.uniform(0, 100, 1000)  # Puedes ajustar el rango y la cantidad de datos

# Calcular la raíz cuadrada para obtener los valores objetivo (y)
y = np.sqrt(X)

# Agregar un poco de ruido para que sea más realista
#y += np.random.normal(0, 1, len(y))

# Visualizar algunos datos generados
print("X:", X[:5])
print("y:", y[:5])


# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

#SVR puede manejar relaciones no lineales mediante el uso de funciones kernel.

model_svr = SVR(kernel='rbf')  # 'rbf' es un kernel radial (gaussiano)
model_svr.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_svr = model_svr.predict(X_test.reshape(-1, 1))

# Evaluar el rendimiento del modelo SVR
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"Mean Squared Error (SVR): {mse_svr}")

# Calcular la raíz cuadrada usando la función del lenguaje para comparar
y_actual = np.sqrt(X_test.flatten())

# Comparar los resultados
comparison = np.column_stack((y_actual, y_pred_svr))
print("Comparación de resultados:")
print("Actual   Predicted")
print(comparison[:10])

