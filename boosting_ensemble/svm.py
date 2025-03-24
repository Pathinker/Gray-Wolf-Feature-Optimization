import sys
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tensorflow.keras.models import load_model   # type: ignore
from tensorflow.keras.models import Model        # type: ignore
from tensorflow.keras.layers import Input        # type: ignore
from sklearn.svm import SVC                      # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.pipeline import Pipeline            # type: ignore
from sklearn.metrics import accuracy_score       # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metaheuristic.custom_layer.mask_layer import MaskLayer

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

neural_network = keras.models.load_model("cnn/alexnet/metaheuristic_alexnet.keras")     # Change path to load other trained model.
neural_network.evaluate(validation_data_frame, verbose = 1)
neural_network.summary()

layer_name = "conv2d"
initial_layer = neural_network.get_layer(layer_name)
layer_name = "mask"     # Change output layer to compare with and without Metaheuristic Optmization.
final_layer = neural_network.get_layer(layer_name)

flatten_neural_network = Model(inputs = initial_layer.input, outputs = final_layer.output)
flatten_neural_network.summary()

def extract_convolution(dataset, neural_network):

    features = []
    labels = []
    
    for images, batch_labels in dataset:
        batch_features = neural_network(images, training=False)
        features.append(batch_features.numpy())
        labels.append(batch_labels.numpy())
    
    return np.concatenate(features), np.concatenate(labels)

x_train, y_train = extract_convolution(train_data_frame, flatten_neural_network)
y_train = y_train.ravel()

SVM = Pipeline([

    ("scaler", StandardScaler()),
    ("svm", SVC(C = 1, kernel = "rbf", gamma = "scale", verbose = True))

])

SVM.fit(x_train, y_train)

save_model = open("boosting_ensemble/svm.sav", "wb")
pickle.dump(SVM, save_model)
save_model.close()

x_validation, y_validation = extract_convolution(validation_data_frame, flatten_neural_network)
y_validation = y_validation.ravel()
y_predict = SVM.predict(x_validation)

accuracy = accuracy_score(y_validation, y_predict)
print(f"SVM Boosting CNN Accuracy: {accuracy}")