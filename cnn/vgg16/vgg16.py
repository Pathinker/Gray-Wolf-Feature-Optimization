import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import regularizers # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore

dataset_path = Path("dataset")
train_path = dataset_path / "train/"
validation_path = dataset_path / "valid/"

image_width = 227
image_height = 227
image_size = [image_width, image_height]
batch_size = 24

train_data_frame = tf.keras.utils.image_dataset_from_directory(
    train_path,
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

validation_data_frame = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

data_argumentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

def apply_data_argumentation(data_frame):

    def process_image(x, y):
        return data_argumentation(x), y

    augmented_data = data_frame.map(process_image)
    labels = np.concatenate([y for x, y in train_data_frame], axis = 0)
    labels = labels.flatten()

    return augmented_data, labels

def balance_clases_dataset(labels):

    class_weight = compute_class_weight(class_weight = "balanced",
                                    classes = np.unique(labels),
                                    y = labels)
    unique_classes = np.unique(labels)
    weights = {}

    for i in range(len(unique_classes)):
        weights[int(unique_classes[i])] = float(class_weight[i])

    return weights

train_data_argumentation, labels = apply_data_argumentation(train_data_frame)
class_weights =  balance_clases_dataset(labels)

VGG16 = keras.models.Sequential([
    keras.layers.Input(shape=(227, 227, 3)),

    # First convolutional layer

    keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Second convolutional layer
    keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Third convolutional layer
    keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Fourth convolutional layer
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Fifth convolutional layer
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = "sigmoid")
    
])

VGG16.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'] 
)

VGG16.summary()

history=VGG16.fit(
    train_data_argumentation,
    epochs=100,
    validation_data=validation_data_frame,
    validation_freq=1,
    class_weight = class_weights
)

VGG16.evaluate(validation_data_frame, verbose = 1)
VGG16.save('cnn/vgg16/vgg16.keras') 

VGG16.history.history.keys()
print('Best validation accuracy score = ',np.max(history.history['val_accuracy']))