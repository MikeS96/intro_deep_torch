# Introduction to neural networks

This folder contains theory and codes related to the fundamentals of neural networks. The main idea is to present these fundamentals in a clear form and provide solution codes. The information presented here is based on the lesson 2 of [Intro to Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188).

## Syllabus
Some of the topics of the course are listed below.

 - What is a neural network?
 - The classification problem
 - The simple perceptron
 - Error functions, maximum likelihood and Cross-Entropy
 - Softmax function
 - Gradient descent
 - Neural networks architecture
 - Multilayer perceptron
 - Overfitting and underfitting
 - Regularization
 - Among others

A detailed explanation of these topics can be seen in the [course notes](https://github.com/MikeS96/intro_deep_torch/tree/master/introduction_neural_networks/Notes) . 

## Codes

The next list shows the codes developed in this section and a brief description of what it does. **Note:** The implementation of the gradient descent and backpropagation algorithms are applied in student admission data.

 - **and_perceptron.py** logistic operator AND implementation with a perceptron.
 - **not_perceptron.py** logistic operator NOT implementation with a perceptron.
 - **cross_entropy.py** Cross-Entropy error function implementation.
 - **perceptron_algorithm.py** Implemented a simple trainable perceptron with logistic regression.
 - **softmax_function.py** Softmax activation function implementation.
 - **GradientDescent.ipynb** Gradient Descent algorithm implemented in a simple perceptron and visualization of the boundary line.
 - **backpropagation.ipynb** Backpropagation implementation in a simple perceptron applied to student admission data.

The result of **GradientDescent.ipynb** can be seen in the next figure where the boundary line of the model has been drawn to show how the perceptron splits the data.

<div  align="center">
<img  src="./Notes/images/boundary_solution.png" width="330">
</div>

