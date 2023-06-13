import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

inp = int(input('Введите величину в градусах: '))

# массивы входных и выходных данных
degrees = np.array([0, 30, 45, 60, 90, 120, 135, 150, 180])
radians = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi])

model = Sequential()  # инициализация последовательной НС
model.add(Dense(20, input_dim=1, activation='relu'))  # добавление скрытого полносвязного слоя
model.add(Dense(1, activation='linear'))  # добавление выходного полносвязного слоя

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(degrees, radians, epochs=2000, batch_size=2, verbose=False)  # обучение

degrees_to_convert = np.array([inp])
radians_predicted = model.predict(degrees_to_convert)  # вычисление радиан

print(degrees_to_convert[0], ' градусов в радианах = ', radians_predicted[0][0], sep='')

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
