# Convolutional Neural Network

<p align="justify">    
Convolutional Neural Networks (CNN) applies filters named kernels on images to extract relevant properties and made predictions with the processed information, this is done as the same procedure of Neural Networks getting a forward propagation method to give a predict, then compute the error by calculating the difference between the expected output and predict one, then backward propagation happens updating their connections known as weights utilize algorithms, this process repeats until some stop criteria or the model was gone through all epochs.
</p>

<p align="justify">    
As mentioned before, CNN are made of multiple layers producing a weighted sum and minimizing error across epochs, nevertheless the following properties modified the output archive better results:
</p>

## Layers

<p align="justify">    
$${\color{orange}- \space Batch \space Normalization:}$$ Shrinks the range of data to get approximately a standard deviation of 1 and variance of 0, this helps the models to train faster and archive better results due to greater sub local starts when weights are first assigned.
</p>

<p align="justify">  
$${\color{orange}- \space Activation:}$$ Since Neural Networks are weighted sum by multiply and sum output for previous layers it produces linear outputs, in other words it could only describe and adjust for problems that could be solve by a straight line to avoid this the data is transform thorough activations layers to modify it and adjust to nonlinear problems.
</p>

<p align="justify">  
$${\color{orange}- \space Max \space Polling:}$$ Max Polling: Once a kernel made a Cross-Correlation product is generated a matrix of values, then a second filter is applied to extract the highest value, this value will represent the most relevant information of the total amount of data. It could also perform an average or lowest value, useful in other types of scenarios.
</p>

<p align="justify">  
$${\color{orange}- \space Flatten:}$$ Transform the output in one dimension vector.
</p>

<p align="justify">  
$${\color{orange}- \space Dense \space Layers:}$$ Fully connected  input neurons to every single output of the next layer, each connection is called weight.
</p>

## Regularizers 

<p align="justify">  
Reduces the overfit by changing how data is generated, flows or loss function is computed.
</p>

<p align="justify">  
$${\color{orange}- \space L2:}$$ Penalize high weight values by adding all the values of the weights to the loss function.
</p>

<p align="justify">  
$${\color{orange}- \space Dropout:}$$ Uses a coefficient to give a chance to disable some neurons on dense layers, this helps to mitigate high dependency on certain neurons and initial weights.
</p>

<p align="justify">  
$${\color{orange}- \space Data \space Argumentation:}$$ Is applied to expand the dataset by adding some mutations to the dataset, changing size, zoom, color or adding noise.
</p>

## Models Tested

<p align="justify">  
$${\color{orange}- \space Alexnet:}$$ Standard CNN archive a notify high score than other models proved deep learning results expenses of it computational calculation, released in 2012.
</p>

<p align="justify">  
$${\color{orange}- \space VGG16:}$$  More robust and weighted model applies consecutive kernels to reduce operations, it principal problem was vanishing gradient, made in 2014.
</p>

<p align="justify">  
$${\color{orange}- \space Resnet101:}$$ Fix vanishing gradient adding residual learning for previous layers this approach is done by establish connections with non-consecutive layers (identity) or by adding an extra convolutional layer (projection), created in 2015.
</p>
