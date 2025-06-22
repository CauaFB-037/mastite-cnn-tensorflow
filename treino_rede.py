import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Caminhos dos diretórios
treino_path = r'G:\Meu Drive\Estudos_programacao\Python\Projeto_Mastite\Data\treino'
validacao_path = r'G:\Meu Drive\Estudos_programacao\Python\Projeto_Mastite\Data\validacao'

# Parâmetros
image_size = (224, 224)
batch_size = 32

# Carregando datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    treino_path,
    image_size=image_size,
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validacao_path,
    image_size=image_size,
    batch_size=batch_size
)

# Normalização
normalization_layer = layers.Rescaling(1./255)

# Aumento de dados (data augmentation)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Aplicar normalização e data augmentation no treino
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(normalization_layer(x), training=True), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Melhorar desempenho com cache e prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Arquitetura da CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binário
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinar modelo
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Visualizar desempenho
plt.plot(history.history['accuracy'], label='Acurácia treino')
plt.plot(history.history['val_accuracy'], label='Acurácia validação')
plt.legend()
plt.title('Desempenho do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()

# Salvar modelo
model.save(r'G:\Meu Drive\Estudos_programacao\Python\Projeto_Mastite\modelo.h5')
