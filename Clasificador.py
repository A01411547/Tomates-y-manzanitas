import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Crear un objeto ImageDataGenerator para realizar data augmentation
data_generator = ImageDataGenerator(
    rescale=1./255,  # Escala los valores de los píxeles al rango [0, 1]
    shear_range=0.2,  # Aplica cortes aleatorios a las imágenes
    zoom_range=0.2,  # Aplica zoom aleatorio a las imágenes
    horizontal_flip=True  # Voltea horizontalmente aleatoriamente las imágenes
)

# Cargar el conjunto de datos de entrenamiento
train_data = data_generator.flow_from_directory(
    'train',
    target_size=(64, 64),  # Cambia el tamaño de las imágenes a 64x64 píxeles
    batch_size=32,
    class_mode='binary'  # Clasificación binaria (tomate o manzana)
)

# Cargar el conjunto de datos de prueba
test_data = data_generator.flow_from_directory(
    'test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Crear un modelo secuencial de TensorFlow
model = Sequential()

# Agregar capas convolucionales para extraer características
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Agregar capas densas para la clasificación
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_data, epochs=10)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_data)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)