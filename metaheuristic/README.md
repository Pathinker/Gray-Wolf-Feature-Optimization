# GWOGPU

<p align="justify">    
Metaheuristic algorithm chooses the most relevant features of processed data in order to improve accuracy inspired on the social hierarchy of grey wolves, takes the best solutions to explore new ones by adding entropy and randomness to find a better combination, unwrap as a dimensional reduction. All the functions require NumPy and TensorFlow data types.
</p>

> [!CAUTION]
> Dimensional reduction requires to select and be at least one flatten layer.

# Documentation

$${\color{violet}GWO.init}$$(

  model, epoch, agents, wolves, class_weight = $${\color{orange}None}$$, transfer_learning = $${\color{orange}None}$$ feature_selection = $${\color{orange}None}$$, ensemble_model = $${\color{orange}None}$$, batch_training = $${\color{orange}None}$$

)

- **model:** Neural network model built on tensorflow to perform the feature selection.
- **agents:** Number of different spaces to search on.
- **wolves:** Number of better solutions storages.
- **class_weight:** Dictionary of values each class has an associated number it is applied on the loss function to balance the samples of the dataset.
- **transfer_learning:** Initializes the spaces of search from weights of a trained model, the weights are randomly multiplied by a value going to 0 - 2.
- **feature_selection:** Name of the layer that is applied the metaheuristic optimization.
- **ensemble_model:** Giving any kind of datatype the exploration process is done with an SVM ensemble model.
- **batch_training:** Trains the ensemble model in batches, specially useful on large models to speed up training, however once it is finished, is recommended to pick up the optimization fund and train it without it to see the true results, due less performance by the distribution of each batch.

$${\color{violet}GWO.get\\_number\\_weights}$$()

<p align="justify">   
Returns the total weights of the model.
</p>

$${\color{violet}GWO.set\\_weights}$$(weights)

<p align="justify"> 
Given a flatten array of weights according to the total amount of weights, reshape and reapply them.
</p>

$${\color{violet}GWO.set\\_position}$$()

<p align="justify"> 
Initializes  the spaces of searching along the upper and lower limits with random and uniform values
</p>

$${\color{violet}GWO.set\\_sele\color{violet}ction}$$() <!-- Somehow github won't add the Latex format if "Select" is written. -->

<p align="justify"> 
Initializes the spaces of searching along the upper and lower limits with random and uniform values, is used sigmoid 
</p>

$${\color{violet}GWO.set\\_mask}$$(mask)

<p align="justify"> 
Given a flatten array of values creates a new object of class mask which makes the feature selection adding a new layer with binary weights, modifying the model and compiling a new one.
</p>

$${\color{violet}GWO.set\\_transfer\\_learning}$$()

<p align="justify"> 
Initializes the spaces of search from weights extracting the weights of the model specified on the constructor, the weights are randomly multiplied by a value going to 0 - 2.
</p>

$${\color{orange}float}$$ $${\color{violet}GWO.weighted\\_loss}$$(<br><br>
	dataset, class_weight<br><br>
)

<p align="justify"> 
Computes the binary cross entropy loss function taking note of the distribution of each class.
</p>

- **dataset:** Keras loaded dataset to apply a predict.
- **class_weight:** Dictionary of values each class has an associated number it is applied on the loss function to balance the samples of the dataset.

$${\color{orange}float}$$ $${\color{violet}GWO.loss\\_features}$$(<br><br>
	train_dataset, validation_dataset, class_weight, epoch<br><br>
)

<p align="justify"> 
Computes the binary cross entropy loss function taking note of the distribution of each class and the amount of features used.
</p>

- **train_dateset:** Keras loaded train dataset to apply a predict.
- **validation_dateset:** Keras loaded validation dataset to apply a predict.
- **class_weight:** Dictionary of values each class has an associated number it is applied on the loss function to balance the samples of the dataset.
- **epoch:** Number of actual epoch.

$${\color{orange}float}$$ $${\color{violet}GWO.loss\\_ensemble}$$(<br><br>
	train_dataset, validation_dataset, class_weight, epoch, batch_training = None<br><br>
)

<p align="justify"> 
Computes the binary cross entropy loss function taking note of the distribution of each class and the amount of features used with a boosting ensemble (SVM).
</p>

- **train_dateset:** Keras loaded train dataset to apply a predict.
- **validation_dateset:** Keras loaded validation dataset to apply a predict.
- **class_weight:** Dictionary of values each class has an associated number it is applied on the loss function to balance the samples of the dataset.
- **epoch:** Number of actual epoch.
- **batch_training:** Enables batch training speeding up training time expenses of performance, oriented to large models.

$${\color{orange}keras\\_model}$$ $${\color{violet}GWO.optimize}$$(<br><br>
	train_dataset, validation_dataset<br><br>
)

<p align="justify"> 
Performs the metaheuristic optimization into the weights of a model.
</p>

- **train_dateset:** Keras loaded train dataset to apply a predict.
- **validation_dateset:** Keras loaded validation dataset to apply a predict.

$${\color{orange}keras\\_model}$$ $${\color{violet}GWO.optimize\\_feature}$$(<br><br>
	train_dataset, validation_dataset, retrain = $${\color{orange}None}$$<br><br>
)

<p align="justify"> 
Performs the metaheuristic optimization choosing the most significant qualities of a model with a train and validation data set.
</p>

- **train_dateset:** Keras loaded train dataset to apply a predict.
- **validation_dateset:** Keras loaded validation dataset to apply a predict.
- **retrain:** Retrain the model to adjust to the dimensionality reduction perform by the metaheuristic optimization.

$${\color{violet}GWO.GWO\\_exploration}$$(<br><br>
	train_dataset, validation_dataset, epoch<br><br>
)

<p align="justify"> 
Computes the explorations of all spaces and updates the wolves according to their loss.
</p>

**train_dateset:** Keras loaded train dataset to apply a predict.
**validation_dateset:** Keras loaded validation dataset to apply a predict.
**epoch:** Number of actual epoch.

$${\color{violet}GWO.GWO\\_feature\\_exploration}$$(<br><br>
	train_dataset, validation_dataset, epoch<br><br>
)

<p align="justify"> 
Computes the explorations of all spaces and updates the wolves according to their loss and amount of features used.
</p>

- **train_dateset:** Keras loaded train dataset to apply a predict.
- **validation_dateset:** Keras loaded validation dataset to apply a predict.
- **epoch:** Number of actual epoch.

$${\color{violet}GWO.update\\_wolves}$$(<br><br>
	loss, accuracy, validation_loss, validation_accuracy, number_features, positions, wolf<br><br>
)

<p align="justify"> 
Modifies the data storage in every wolf when a better solution is found.
</p>

- **loss:** Loss number
- **accuracy:** Accuracy number
- **validation_loss:** Loss number of the validation dataset
- **validation_accuracy:** Accuracy number of the validation dataset.
- **number_features:** Amount of features used.
- **positions:** Associated weights with the performance.
- **wolf:** Index number of the wolf desired to change values.

$${\color{violet}GWO.get\\_report}$$()

<p align="justify"> 
Export in a csv file all the performance across every single epoch and the best weights in txt format.
</p>

$${\color{violet}GWO.GWO\\_explotation}$$(epoch)

<p align="justify"> 
Computes the GWO algorithm to find a high quality solution taking notes of the best previous solutions with some randomness using pycuda to parallelize code.
</p>

- **epoch:** Number of actual epoch.

$${\color{violet}GWO.GWO\\_feature\\_exploration}$$(epoch)

<p align="justify"> 
Computes the GWO algorithm to find a high quality combination of features taking notes of the best previous solutions with some randomness using pycuda to parallelize code.
</p>

- **epoch:** Number of actual epoch.

$${\color{orange}int}$$ $${\color{violet}GWO.get\\_seed}$$()

<p align="justify"> 
Returns a random number generated by entropy of the operative system, current time and a random number of standard python libraries.
</p>
