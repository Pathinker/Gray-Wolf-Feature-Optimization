# Suport Vector Machines

<p align="justify">
A support vector machine draws a line, plane or hyperplane according to the dimensions denoted by the number of variables, generating divisions or segments in subspaces where all the stored values ​​are incorporated into the same classification. When evaluating new data, pertinent calculations are performed and the prediction is made based on the corresponding subspace.
</p>

<p align="justify">
In order to generate this division, support vectors are taken into consideration. The data used as support vectors correspond to those closest to other classifications, creating a margin and, consequently, the subspaces. The margin is permuted by taking into account the maximum separation between the support vectors, generating a maximal margin classifier. In some circumstances, depending on the attributes of the training data, they contain high variation and, consequently, the presence of records at the extremes of the classifications, resulting in bias by generating an ineffective limit or threshold.
</p>

<p align="justify">
By applying miss classification, neglecting or omitting atypical magnitudes improves the accuracy of the model. This miss classification occurs with cross-validation, where different segments of the entire dataset are taken into account to randomize the validation sampling. Implementing miss classification is certified to achieve higher overall performance at the expense of decreasing training accuracy.
</p>

## Kernels

<p align="justify">
Support Vector Machines could use different methods to get the hyperplane allowing to identify different properties on data by looking at the archiving results.
</p>

<p align="justify">
$${\color{orange}- \space Linear \space Kernel:}$$ Performs scalar product between vectors to separate data linearly.
</p>

<p align="justify">
$${\color{orange}- \space Polynomial \space Kernel:}$$ It uses a kernel taking into consideration two coefficients "r" conditioning the result of the dot product obtained by raising the value, the value of the coefficients is disregarded and "d" which consolidates the degree of the polynomial, both coefficients are obtained by performing a cross-validation.
</p>

<p align="justify">
When obtaining the coefficients, a dot or vector product is obtained that is equivalent to the expansion of the polynomial. The vector product produces a perpendicular vector that is not inclined to any of the planes, which is used to create subspaces with the maximum possible margin. To evaluate the records with respect to the dot product, the polynomial kernel is used, since they are equivalent, saving the computational operations required to calculate the dot product of each of the records and their high-dimensional relationship.
</p>

<p align="justify">
$${\color{orange}- \space Radial \space Kernel:}$$ Allows infinite scaling in dimensions, exponentially evaluates a difference of squares that is multiplied by a coefficient acquired by cross-validation, when computing the values ​​it returns the impact of the nearby values ​​interacting in a way that identifies a nearest neighbor weight where the nearby records have a greater influence on the classification.
</p>

<p align="justify">
It is capable of demonstrating the radial kernel formula as the result of infinitely evaluating and transforming data using the Taylor series, expressing the result of a function as the infinite summation of functions while the value exists.
</p>
