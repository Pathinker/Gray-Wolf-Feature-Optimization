import sys
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metaheuristic.gwogpu import GWO

dataset_path = Path("dataset")
train_path = dataset_path / "train/"
validation_path = dataset_path / "valid/"

anchoImagen = 227
largoImagen = 227
image_size = [anchoImagen, largoImagen]
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

model_name = "alexnet"      # Change name to train other gradient models.
model = keras.models.load_model(f"cnn/{model_name}/{model_name}.keras")
    
gwo = GWO(model=model, epochs=100, agents= 10, wolves = 10, class_weight = class_weights, feature_selection = "flatten", ensemble_model = True)
model = gwo.optimize_feature(train_data_argumentation, validation_data_frame, retrain = True)
model.save(f"cnn/{model_name}/metaheuristic_{model_name}.keras")
shutil.move("cnn/weights.txt", f"cnn/{model_name}/weights.txt")
shutil.move("cnn/report.csv", f"cnn/{model_name}/report.csv")