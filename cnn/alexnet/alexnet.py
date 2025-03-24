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

alexnet = keras.models.Sequential([
    keras.layers.Input(shape=(227, 227, 3)),

    # First convolutional layer 96 Kernels of (11, 11)
    keras.layers.Conv2D(filters = 96, kernel_size = (11, 11),strides = (4, 4), kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = "valid", data_format = None),

    # Segunda convolutional layer 256 Kernels of (5, 5)
    keras.layers.Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), padding = "same", kernel_initializer = "he_normal", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = "valid", data_format = None),

    # Third convolutional layer 384 Kernels of (3, 3)
    keras.layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = "same", kernel_initializer = "he_normal", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),

    # Fourth convolutional layer 384 of Kernels (3, 3)
    keras.layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = "same", kernel_initializer = "he_normal", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),

    # Fifth convolutional layer 256 of Kernels (3, 3)
    keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same", kernel_initializer = "he_normal", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2),  padding = "valid",data_format = None),

    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation = "relu", kernel_regularizer=regularizers.l2(0.0001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = "sigmoid")
])

alexnet.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'] 
)

alexnet.summary()

history=alexnet.fit(
    train_data_argumentation,
    epochs=100,
    validation_data=validation_data_frame,
    validation_freq=1,
    class_weight = class_weights
)

alexnet.evaluate(validation_data_frame, verbose = 1)
alexnet.save('cnn/alexnet/alexnet.keras') 

alexnet.history.history.keys()
print('Best validation accuracy score = ',np.max(history.history['val_accuracy']))