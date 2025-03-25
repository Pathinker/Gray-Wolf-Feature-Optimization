import os
import csv
import time
import hashlib
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable          # type: ignore
from tensorflow.keras.layers import Layer, Flatten, Dense, Input        # type: ignore
from tensorflow.keras.models import load_model, Model                   # type: ignore

import pycuda.autoinit                                                  # type: ignore
import pycuda.driver as cuda                                            # type: ignore
from pycuda.compiler import SourceModule                                # type: ignore

from sklearn.svm import SVC                                             # type: ignore
from sklearn.preprocessing import StandardScaler                        # type: ignore
from sklearn.pipeline import Pipeline                                   # type: ignore
from sklearn.metrics import accuracy_score                              # type: ignore

from metaheuristic.custom_layer.mask_layer import MaskLayer

class GWO:
    def __init__(self, model, epochs=10, agents = 5,  wolves = 5, class_weight = None, transfer_learning = None, feature_selection = None, ensemble_model = None, batch_training = None):

        self.model = model
        self.ORIGINAL_MODEL = model
        self.epochs = epochs
        self.agents = agents
        self.wolves = wolves
        self.wolves_name = {0 : "Alfa", 1 : "Beta", 2: "Delta"}
        self.class_weight = class_weight
        self.transfer_learning = transfer_learning
        self.feature_selection = feature_selection
        self.ensemble_model = ensemble_model
        self.batch_training = batch_training
        self.upper_bound = 0.5
        self.lower_bound = -0.5
        self.alfa = 0.99
        self.beta = 0.01

        self.features = None
        self.number_features = None
        self.weights_structure = None
        self.number_weights = None

        for wolf in range(self.wolves - 3):
            self.wolves_name[wolf + 3] = (f"Omega {wolf + 1}")

        if(feature_selection is None):
            self.weights_structure = model.get_weights()
            weights = self.get_number_weights()
            self.number_weights = np.uint32(weights)
        else:
            self.features = self.model.get_layer(feature_selection)
            self.input_features = self.features.get_build_config()
            self.input_shape = self.input_features["input_shape"]
            weights = [dim for dim in self.input_shape if dim is not None]

            self.number_weights = 1

            for value in weights:
                self.number_weights *= value

            self.number_weights = np.uint32(self.number_weights)

        self.positions = np.zeros((agents, self.number_weights))
        self.round_positions = np.zeros((agents, self.number_weights))
        self.number_features = np.zeros((agents))
        self.wolves_positions = np.zeros((self.wolves, self.number_weights))
        self.number_features_wolves = np.zeros((self.wolves))

        self.loss = np.full(self.wolves, np.finfo(np.float64).max)
        self.accuracy = np.zeros((self.wolves))
        self.validation_loss = np.full(self.wolves, np.finfo(np.float64).max)
        self.validation_accuracy = np.zeros((self.wolves))
        
        self.loss_log = np.zeros((self.wolves, self.epochs))
        self.accuracy_log = np.zeros((self.wolves, self.epochs))
        self.validation_loss_log = np.zeros((self.wolves, self.epochs))
        self.validation_accuracy_log = np.zeros((self.wolves, self.epochs))
        self.number_features_log = np.zeros((self.wolves, self.epochs))
        
        if(feature_selection is not None):
            self.set_selection()
            return      # Feature Selection Only is not necesary to train whole model.

        if(transfer_learning is not None):
            self.set_transfer_learning()
        else:
            self.set_position()
        
        self.set_weights(self.round_positions[0])

    def get_number_weights(self):

        total_weights = 0

        for weights in self.weights_structure:
            elementos = np.prod(weights.shape)
            total_weights += elementos

        return total_weights

    def set_weights(self, weights):

        new_weights = []
        index = 0

        for layer_weights in self.weights_structure:
            shape = layer_weights.shape
            size = np.prod(shape)
            new_weights.append(np.array(weights[index:index + size]).reshape(shape))
            index += size

        self.model.set_weights(new_weights)

    def set_position(self):

        for i in range(self.agents):

            random_position = []

            for weights in self.weights_structure:
                random_weights = np.random.uniform(self.lower_bound, self.upper_bound, weights.shape)
                random_position.append(random_weights.flatten())

            self.round_positions[i] = np.concatenate(random_position)
    
    def set_selection(self):

        random_number = np.random.random(self.agents)
        random_signed = np.random.random(self.agents)
        average_limiter = ((np.fabs(self.lower_bound) + np.fabs(self.upper_bound)) / 2)

        for i in range(len(random_number)):
            random_number[i] *= average_limiter          

            if(random_signed[i] > 0.5):
                random_number[i] *= -1

        for i in range(self.agents):
            position = []
            round_position = []
            number_features = 0

            for j in range(self.number_weights):
                if(random_number[i] > 0):
                    random_weights = np.random.uniform(self.lower_bound + random_number[i], self.upper_bound)
                else:
                    random_weights = np.random.uniform(self.lower_bound, self.upper_bound + random_number[i])
                
                position.append(random_weights)

            for j in range(self.number_weights):
                sigmoid = 1 / (1 + np.exp(position[j]))

                if(sigmoid > 0.5):
                    number_features += 1
                    round_position.append(1)
                else:
                    round_position.append(0)
            
            self.round_positions[i] = np.array(round_position) 
            self.positions[i] = np.array(position)
            self.number_features[i] = number_features

    def set_mask(self, mask):

        entry_layer = self.ORIGINAL_MODEL.get_layer("conv2d").input
        output_layer = self.ORIGINAL_MODEL.get_layer(self.feature_selection).output

        mask_layer = MaskLayer(mask=mask)(output_layer)
        output_layer = mask_layer

        add_dense_layers = False
        reshape_next_mask_layer = False

        for layer in self.ORIGINAL_MODEL.layers:
            if (add_dense_layers is False):
                if(layer.name == 'flatten'):
                    add_dense_layers = True
                continue

            if(reshape_next_mask_layer is False):
                weights, biases = layer.get_weights()
                indices_to_keep = np.where(np.array(mask) == 1)[0]

                new_weights = weights[indices_to_keep, :].copy()
                new_biases = biases.copy()

                new_layer = layer.__class__.from_config(layer.get_config())
                new_layer.build(input_shape=(None, int(mask.sum())))
                new_layer.set_weights([new_weights, new_biases])

                layer = new_layer                
                reshape_next_mask_layer = True

            output_layer = layer(output_layer)
                
        custom_model = Model(inputs=entry_layer, outputs=output_layer)

        custom_model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )

        self.model = custom_model

    def set_transfer_learning(self):

        self.transfer_learning= self.model.get_weights()
        flattened_weights = np.concatenate([weight.flatten() for weight in self.transfer_learning])
        self.transfer_learning = flattened_weights

        for i in range(self.agents):
            inicial_time = time.time()
            weights = len(self.transfer_learning)

            positions = np.array(self.round_positions[i], dtype=np.float32)
            transfer_learning = np.array(self.transfer_learning, dtype=np.float32)

            positions_distance = cuda.mem_alloc(positions.nbytes)
            cuda.memcpy_htod(positions_distance, positions)

            transfer_learning = cuda.mem_alloc(self.transfer_learning.nbytes)
            cuda.memcpy_htod(transfer_learning, self.transfer_learning)

            mod = SourceModule("""

            __device__ float xorshift32(unsigned int seed) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF) * 2;
            }
            __global__ void update(
                float *positions,
                float *transfer_learning,
                int weights_number,
                float lower_bound,
                float upper_bound,
                unsigned int seed
            ){
                int thread = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (thread < weights_number) {
             
                    unsigned int thread_seed = seed + thread;
                    float random = xorshift32(thread_seed);
            
                    positions[thread] = transfer_learning[thread] * random;
                               
                    if(positions[thread] < lower_bound) {
                        positions[thread] = lower_bound;            
                    } else if(positions[thread] > upper_bound) {       
                        positions[thread] = upper_bound;
                    }               
                }
            }
            """)

            update_positions = mod.get_function("update")
            block = 1024
            grid = (weights + block - 1) // block
            seed = self.get_seed()

            update_positions(
                            positions_distance,
                            transfer_learning, 
                            np.int32(weights), 
                            np.float32(self.lower_bound), 
                            np.float32(self.upper_bound),
                            seed, 
                            block=(block, 1, 1),
                            grid=(grid, 1)
                            )

            cuda.memcpy_dtoh(positions, positions_distance)
            self.round_positions[i] = positions

            final_time = time.time()
            print(f"Initialization {i + 1} / {self.agents} Execution time: {final_time - inicial_time} seconds")

    def weighted_loss(self, dataset, class_weight):

        loss = 0.0
        accuracy = 0.0
        total = 0
        
        for x, y in tqdm(dataset, desc = f"Calculating Loss", unit="batch"):
            weighted_losses = []

            for label in y:
                weight = class_weight[int(label)]
                weighted_losses.append(weight)

            weighted_losses = np.array(weighted_losses)
            
            loss_batch, accuracy_batch = self.model.evaluate(x, y, sample_weight = weighted_losses, verbose = 0)

            loss += loss_batch
            accuracy += accuracy_batch
            total += 1

        print(f"Loss: {loss/total}, Accuracy = {accuracy / total}")
        return (loss / total), (accuracy / total)

    def loss_features(self, train_dataset, validation_dataset, class_weight, epoch):

        loss = 0.0
        accuracy = 0.0
        total = 0

        alfa = 0.99
        beta = 0.01
        
        for x, y in tqdm(train_dataset, desc = f"Calculating Loss", unit="batch"):
            weighted_losses = []

            for i in y:
                peso = class_weight[int(i)]
                weighted_losses.append(peso)

            weighted_losses = np.array(weighted_losses)
            
            loss_batch, accuracy_batch = self.model.evaluate(x, y, sample_weight = weighted_losses, verbose = 0)

            loss += loss_batch
            accuracy += accuracy_batch
            total += 1
        
        loss = alfa * loss + beta * (self.number_features[epoch] / self.number_weights)
        validation_loss, validation_accuracy = self.model.evaluate(validation_dataset, verbose=1)    

        print(
            f"Loss: {loss / total}, "
            f"Accuracy: {accuracy / total}, "
            f"Validation Loss: {validation_loss}, "
            f"Validation Accuracy: {validation_accuracy}, "
            f"Features: {self.number_features[epoch]}"
        )
                
        return (loss / total), (accuracy / total), validation_loss, validation_accuracy, self.number_features[epoch]

    def loss_ensemble(self, train_dataset, validation_dataset, class_weight, epoch, batch_training = None):

        inicial_time = time.time()

        alfa = self.alfa
        beta = self.beta

        layer_name = "conv2d"
        entry_layer = self.model.get_layer(layer_name)

        layer_name = "mask"  
        output_layer = self.model.get_layer(layer_name)

        flatten_model = Model(inputs = entry_layer.input, outputs = output_layer.output)

        svm = Pipeline([

            ("scaler", StandardScaler()),
            ("svm", SVC(C = 1, kernel = "rbf", gamma = "scale", verbose = batch_training is None))

        ])

        def get_svm_dataset(dataset, model):
  
            features = []
            labels = []

            for images, tag in dataset:
                batch_features, batch_labels = get_convolution(images, tag, model)
                features.append(batch_features)
                labels.append(batch_labels)
                  
            x = np.concatenate(features)
            y = np.concatenate(labels)
            y = y.ravel()

            return x, y
        
        def get_convolution(x, y, model):

            convolution = model(x, training = False)
            features = convolution.numpy()
            labels = y.numpy()

            return features, labels

        x_train, y_train = get_svm_dataset(train_dataset, flatten_model)
        x_validation, y_validation = get_svm_dataset(validation_dataset, flatten_model)

        if(batch_training is None):
            svm.fit(x_train, y_train)
        else:
            for images, tag in tqdm(train_dataset, desc = "Training SVM", unit = "batch"):
                features, labels = get_convolution(images, tag, flatten_model)
                labels = labels.ravel()

                if(len(np.unique(labels)) > 1):
                    svm.fit(features, labels)

        def get_loss(x, y, svm, class_weight):

            predict = svm.predict(x)
            decision_function = svm.decision_function(x)
            probabilities = 1 / (1 + np.exp(-decision_function)) 

            binary_crossentropy = tf.keras.losses.binary_crossentropy(y, probabilities)
            losses_list = []

            for label in y:

                loss = class_weight[label]
                losses_list.append(loss)

            losses_list = np.array(losses_list)
            weighted_loss = binary_crossentropy * losses_list
            weighted_loss = np.mean(weighted_loss)
            
            correct_predictions = []

            for estimate, expected in zip(predict, y):
                if(estimate > 0.5):
                    estimate = 1
                else:
                    estimate = 0

                if(estimate == expected):
                    correct_predictions.append(1)
                else:
                    correct_predictions.append(0)
            
            accuracy = sum(correct_predictions) / len(correct_predictions)
            weighted_loss = alfa * weighted_loss + beta * (self.number_features[epoch] / self.number_weights)
            
            return weighted_loss, accuracy
        
        train_loss, train_accuracy = get_loss(x_train, y_train, svm, class_weight)
        validation_loss, validation_accuracy = get_loss(x_validation, y_validation, svm, class_weight)

        final_time = time.time()

        print(f"Execution Time: {final_time - inicial_time} seconds")
        print(
            f"Loss: {train_loss}, "
            f"Accuracy: {train_accuracy}, "
            f"Validation Loss: {validation_loss}, "
            f"Validation Accuracy: {validation_accuracy}, "
            f"Features:  {self.number_features[epoch]}"
            )

        return train_loss, train_accuracy, validation_loss, validation_accuracy,  self.number_features[epoch] 

    def optimize(self, train_dataset, validation_dataset):

        for epoch in range(self.epochs):
            self.GWO_exploration(train_dataset, validation_dataset, epoch)
            self.GWO_explotation(epoch)

            for i in range(self.wolves):
                self.accuracy_log[i][epoch] = self.accuracy[i]
                self.loss_log[i][epoch] = self.loss[i]
                self.validation_accuracy_log[i][epoch] = self.validation_accuracy[i]
                self.validation_loss_log[i][epoch] = self.validation_loss[i]
        
        self.set_weights(self.wolves_positions[0])
        self.get_report()
        return self.model
    
    def optimize_feature(self, train_dataset, validation_dataset, retrain = None):

        for epoch in range(self.epochs):
            self.GWO_feature_exploration(train_dataset, validation_dataset, epoch)
            self.GWO_feature_explotation(epoch)

            for i in range(self.wolves):
                self.accuracy_log[i][epoch] = self.accuracy[i]
                self.loss_log[i][epoch] = self.loss[i]
                self.validation_accuracy_log[i][epoch] = self.validation_accuracy[i]
                self.validation_loss_log[i][epoch] = self.validation_loss[i] 
                self.number_features_log[i][epoch] = self.number_features_wolves[i]           

        for i in range(len(self.wolves_positions[0])):

            if(self.wolves_positions[0, i] > 0.5):
                self.wolves_positions[0, i] = 1
            else:
                self.wolves_positions[0, i] = 0

        self.set_mask(self.wolves_positions[0])
        self.get_report()

        if(retrain is None):
             return self.model

        freeze_training = True

        for layer in self.model.layers:
            if(layer.name == "mask"):
                freeze_training = False
                continue

            if(freeze_training is True):
                layer.trainable = False
            else:
                layer.trainable = True 

        self.model.fit(
                       train_dataset,
                       epochs=10,
                       validation_data=validation_dataset,
                       validation_freq=1,
                       class_weight = self.class_weight
                      )       

        return self.model
    
    def GWO_exploration(self, train_dataset, validation_dataset, epoch):

        for n in range(self.agents):
            print(f"Exploration Epoch {epoch + 1} / {self.epochs} (Agent {n + 1} / {self.agents})| Train | Validation: ")

            self.set_weights(self.round_positions[n])
            loss, accuracy = self.weighted_loss(train_dataset, self.class_weight)
            validation_loss, validation_accuracy = self.model.evaluate(validation_dataset, verbose=1)    

            for i in range(self.wolves):
                if(loss < self.loss[i]):
                    print(f"{self.wolves_name[i]} Update")
                    self.update_wolves(loss, accuracy, validation_loss, validation_accuracy, i, np.ravel(self.round_positions[n, :].copy()))
                    break
            
            for i in range(self.wolves):
                print(
                    f"{self.wolves_name[i]} -> "
                    f"Loss: {self.loss[i]}, "
                    f"Accuracy: {self.accuracy[i]}, "
                    f"Validation_Loss: {self.validation_loss[i]}, "
                    f"Validation_Accuracy: {self.validation_accuracy[i]}"
                    )
    
    def GWO_feature_exploration(self, train_dataset, validation_dataset, epoch):

        for n in range(self.agents):
            print(f"Exploration Epoch {epoch + 1} / {self.epochs} (Agent {n + 1} / {self.agents})| Train | Validation: ")

            self.set_mask(self.round_positions[n])

            if(self.ensemble_model is None):
                loss, accuracy, validation_loss, validation_accuracy, number_features =  self.loss_features(train_dataset, validation_dataset, self.class_weight, n)
            else:
                loss, accuracy, validation_loss, validation_accuracy, number_features = self.loss_ensemble(train_dataset, validation_dataset, self.class_weight, n, self.batch_training)

            for i in range(self.wolves):
                if(loss < self.loss[i]):
                    print(f"{self.wolves_name[i]} Update")
                    self.update_wolves(loss, accuracy, validation_loss, validation_accuracy, i, np.ravel(self.round_positions[n, :].copy()), number_features)
                    break
            
            for i in range(self.wolves):
                print(
                    f"{self.wolves_name[i]} -> "
                    f"Loss: {self.loss[i]}, "
                    f"Accuracy: {self.accuracy[i]}, "
                    f"Validation_Loss: {self.validation_loss[i]}, "
                    f"Validation_Accuracy: {self.validation_accuracy[i]}, "
                    f"Features: {self.number_features_wolves[i]}"
                    )
        
    def update_wolves(self, loss, accuracy, validation_loss, validation_accuracy, wolf, positions = None, number_features = None):

        for i in range(self.wolves - 1, wolf - 1, -1):
            if(i == wolf):
                self.loss[i] = loss
                self.accuracy[i] = accuracy
                self.validation_loss[i] = validation_loss
                self.validation_accuracy[i] = validation_accuracy
                self.wolves_positions[i] = positions
                self.number_features_wolves[i] = number_features
            else:
                self.loss[i] = self.loss[i - 1]
                self.accuracy[i] = self.accuracy[i - 1]
                self.validation_loss[i] = self.validation_loss[i - 1]
                self.validation_accuracy[i] = self.validation_accuracy[i - 1] 
                self.wolves_positions[i] = self.wolves_positions[i - 1]
                self.number_features_wolves[i] = self.number_features_wolves[i - 1]     

    def get_report(self):

        with open('cnn/report.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Wolf', "Epoch" 'Train Accuracy', 'Train Loss', 'Validation Accuracy', 'Validation Loss']

            if self.feature_selection is not None:
                 header.append('Number of Features')
            writer.writerow(header)
    
            for i in range(self.wolves):
                for epoch in range(self.epochs):
                    row = [
                        self.wolves_name[i],
                        epoch + 1,
                        self.accuracy_log[i][epoch],
                        self.loss_log[i][epoch],
                        self.validation_accuracy_log[i][epoch],
                        self.validation_loss_log[i][epoch]
                    ]
                    if self.feature_selection is not None:
                        row.append(self.number_features_log[i][epoch])
                    
                    writer.writerow(row)

        with open('cnn/weights.txt', 'w') as f:

            for i in self.wolves_positions[0]:
                f.write(str(i) + ',')  

    def GWO_explotation(self, epoch):

        for i in range(self.agents):
            inicial_time = time.time()
            weights = len(self.round_positions[0])

            a = 2 - epoch * (2 / self.epochs)

            positions = np.array(self.round_positions[i], dtype=np.float32)
            posicion = np.array(self.wolves_positions, dtype=np.float32)

            positions_distance = cuda.mem_alloc(positions.nbytes)
            cuda.memcpy_htod(positions_distance, positions)

            positions_distance_wolves= cuda.mem_alloc(posicion.nbytes)
            cuda.memcpy_htod(positions_distance_wolves, posicion)

            mod = SourceModule("""
            #define MAXWOLVES """ + str(self.wolves) + """

            __device__ float xorshift32(unsigned int seed) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF); // Normalizar a rango [0, 1]
            }
            __global__ void update(
                float *positions,
                float *wolves_positions,
                float a, int weights_number,
                float lower_bound, 
                float upper_bound,  
                unsigned int seed
            ){
                                   
                int thread = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (thread < weights_number) {
                               
                    float A[MAXWOLVES];
                    float C[MAXWOLVES];
                    float next_position[MAXWOLVES];
                    float solution = 0.0;
                    unsigned int thread_seed = seed + thread;
                               
                    for (int i = 0; i < MAXWOLVES; i++) {

                        float r1 = xorshift32(thread_seed + i);
                        float r2 = xorshift32(thread_seed + i + MAXWOLVES);
                               
                        A[i] = 2 * a * r1 - a; // (2 * a - a) * ()
                        C[i] = 2 * r2;
                    }   
                               
                    for (int i = 0; i < MAXWOLVES; i++) {
                               
                        float current_wolf_position = wolves_positions[thread + (weights_number * i)];
                        float prey_position = positions[thread];

                        next_position[i] = fabs(C[i] * current_wolf_position - prey_position);
                        float X = current_wolf_position - A[i] * next_position[i];
                        solution += X;
                    }
                               
                    solution /= MAXWOLVES;
                               
                    positions[thread] = solution;
                               
                    if(positions[thread] < lower_bound) {
                        positions[thread] = lower_bound;            
                    } else if(positions[thread] > upper_bound) {
                        positions[thread] = upper_bound;
                    }               
                }
            }
            """)

            update_positions = mod.get_function("update")
            block = 1024
            grid = (weights + block - 1) // block
            seed = self.get_seed()

            update_positions(
                            positions_distance,
                            positions_distance_wolves,
                            np.float32(a),
                            np.int32(weights),
                            np.float32(self.lower_bound), 
                            np.float32(self.upper_bound),
                            seed, block=(block, 1, 1),
                            grid=(grid, 1)
                            )

            cuda.memcpy_dtoh(positions, positions_distance)
            self.round_positions[i] = positions

            final_time = time.time()
            print(f"Explotation {i + 1} / {self.agents} Execution time: {final_time - inicial_time} seconds")

    def GWO_feature_explotation(self, epoch):

            for i in range(self.agents):
                inicial_time = time.time()
                weights = len(self.round_positions[0])

                a = 2 - epoch * (2 / self.epochs)

                positions = np.array(self.round_positions[i], dtype=np.float32)
                loss = np.array(self.loss, dtype=np.float32)
                round_positions = np.array(self.round_positions[i], dtype=np.float32)
                posicion = np.array(self.wolves_positions, dtype=np.float32)

                total_loss = sum(self.loss)

                for i in range(self.wolves):
                    loss[i] = self.loss[i] / total_loss

                positions_distance = cuda.mem_alloc(positions.nbytes)
                cuda.memcpy_htod(positions_distance, positions)

                normalized_loss = cuda.mem_alloc(self.loss.nbytes)
                cuda.memcpy_htod(normalized_loss, loss)

                positions_distance_round = cuda.mem_alloc(round_positions.nbytes)
                cuda.memcpy_htod(positions_distance_round, round_positions)

                positions_distance_wolves= cuda.mem_alloc(posicion.nbytes)
                cuda.memcpy_htod(positions_distance_wolves, posicion)

                mod = SourceModule("""
                #define MAXWOLVES """ + str(self.wolves) + """

                __device__ float xorshift32(unsigned int seed) {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    return (seed & 0x7FFFFFFF) / float(0x7FFFFFFF); // Normalizar a rango [0, 1]
                }
                __global__ void update(
                    float *round_positions, 
                    float *positions,
                    float *wolves_positions, 
                    float *loss, 
                    float a,
                    int weights_number, 
                    float lower_bound, 
                    float upper_bound, 
                    unsigned int seed,
                    int epoch,
                    int max_epochs
                ){

                    int thread = blockIdx.x * blockDim.x + threadIdx.x;
                    
                    if (thread < weights_number) {
                                
                        float A[MAXWOLVES];
                        float C[MAXWOLVES];
                        float next_position[MAXWOLVES];
                        float solution = 0.0;
                        unsigned int thread_seed = seed + thread;    
                        
                        for (int i = 0; i < MAXWOLVES; i++){

                            unsigned int seed1 = thread_seed ^ (i * 2654435761U) ^ (thread_seed >> 13);
                            unsigned int seed2 = thread_seed ^ ((i + MAXWOLVES) * 2246822519U) ^ (thread_seed << 7);
                        
                            float r1 = xorshift32(seed1);
                            float r2 = xorshift32(seed2);
                                
                            A[i] = 2 * a * r1 - a;
                            C[i] = 2 * r2;
                        }   
                                
                        for (int i = 0; i < MAXWOLVES; i++){
                                
                            float current_wolf_position = wolves_positions[thread + (weights_number * i)];
                            float prey_position = positions[thread];

                            next_position[i] = fabs(C[i] * current_wolf_position - prey_position);
                            float X = current_wolf_position - A[i] * next_position[i];
                            X *= loss[i];
                            solution += X;
                        }

                        solution /= MAXWOLVES;

                        float r1 =  a - epoch * (a / max_epochs);
                        float r2 = xorshift32(thread_seed ^ (374761393U) ^ (thread_seed << 11));
                        float r3 = xorshift32(thread_seed ^ (217645177U) ^ (thread_seed >> 11));
                        float r4 = xorshift32(thread_seed ^ (2654435761U) ^ (thread_seed << 13));

                        if(r4 < 0.5) {
                            solution += (r1 * sinf(r2) * fabs(r3 * wolves_positions[thread] - solution));
                        } else {
                            solution += (r1 * cosf(r2) * fabs(r3 * wolves_positions[thread] - solution));
                        }

                        positions[thread] = 1 / (1 + exp(-solution));

                        if(positions[thread] < lower_bound) {
                            positions[thread] = lower_bound;            
                        } else if(positions[thread] > upper_bound){      
                            positions[thread] = upper_bound;
                        }   

                        round_positions[thread] = positions[thread];
         
                        if (round_positions[thread] < 0.5) {
                            round_positions[thread] = 0;
                        } else {
                            round_positions[thread] = 1;
                        }                     
                    }
                }
                """)

                update_positions = mod.get_function("update")
                block = 1024
                grid = (weights + block - 1) // block

                seed = self.get_seed()

                update_positions(
                                positions_distance_round,
                                positions_distance,
                                positions_distance_wolves,
                                normalized_loss,
                                np.float32(a), 
                                np.int32(weights), 
                                np.float32(self.lower_bound),
                                np.float32(self.upper_bound),
                                seed, 
                                np.int32(epoch),
                                np.int32(self.epochs),
                                block=(block, 1, 1),
                                grid=(grid, 1)
                                )

                cuda.memcpy_dtoh(positions, positions_distance)
                cuda.memcpy_dtoh(round_positions, positions_distance_round)

                self.round_positions[i] = round_positions
                self.positions[i] = positions

                number_features = 0

                for feature in self.round_positions[i]:
                    if(feature == 1):
                        number_features += 1

                self.number_features[i] = number_features

                final_time = time.time()
                print(f"Explotation {i + 1} / {self.agents} Execution time: {final_time - inicial_time} seconds.")

    def get_seed(self):

        operative_sistem_entropy = int.from_bytes(os.urandom(4), byteorder='big')
        
        current_time = time.time_ns()
        hash_time = int(hashlib.sha256(str(current_time).encode()).hexdigest(), 16)
        
        random_number = np.random.randint(0, 2**32, dtype=np.uint32)
        
        seed = (operative_sistem_entropy ^ hash_time ^ random_number) % (2**32)
        return np.uint32(seed)