import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar los datos desde el archivo CSV
data = pd.read_csv('oil.csv', delimiter=';')

# Seleccionar las variables de entrada (X) y la variable de salida (y)
X = data[['Close', 'Var', 'Maximo', 'Minimo']].values
y = data['Close'].shift(-1).values[:-1]  # El valor del petróleo del día siguiente

# Normalizar los datos
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-1], y_scaled, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(Dense(1, activation='linear'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Loss en el conjunto de prueba: {loss}')

# Hacer predicciones para el día siguiente
last_data = X_scaled[-1].reshape(1, -1)
predicted_scaled = model.predict(last_data)
predicted = scaler_y.inverse_transform(predicted_scaled)

print(f'Predicción del valor del petróleo del día siguiente: {predicted[0, 0]}')
