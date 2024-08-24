# Deep Learning 

![License: MIT](https://img.shields.io/badge/license-MIT-red?style=plastic)
![Python](https://img.shields.io/badge/python-blue?style=plastic&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-white?style=plastic&logo=tensorflow)
![NumPy](https://img.shields.io/badge/numpy-midnightblue?style=plastic&logo=NumPy)




Welcome to my Deep Learning Repository! This repository showcases the projects, tasks, and materials I've completed during my deep learning journey, specifically through the **Deep Learning Specialization** offered by [DeepLearning.AI](https://www.deeplearning.ai) on [Coursera](https://www.coursera.org/).

### [Deep Learning Specialization](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/certificates/deep_learning_specialization.pdf)

This specialization consists of five fundamental courses:

- [Neural Networks and Deep Learning](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/certificates/neural_networks_%26_deep_learning.pdf)
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization, and Optimization](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/certificates/improving_deep_neural_network.pdf)
- [Structuring Machine Learning Projects](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/certificates/structuring_ml_projects.pdf)
- [Convolutional Neural Networks](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/certificates/conv_nets.pdf)
- [Sequence Models](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/certificates/sequence_models.pdf)

### Models

In the [**models**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/certificates/sequence_models.pdf) directory, you'll find implementations of over 15 deep learning models that I developed throughout my learning journey. These include:

- Deep Multi-Layer Perceptron (MLP) models
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) networks
- Transformer models

Each model is implemented from scratch, with some also utilizing TensorFlow. Additionally, I have experimented with various optimization algorithms such as Gradient Descent (GD) and Adam, also implemented from scratch.

Feel free to explore the models, and consider adapting them to your own projects and datasets.

| Model                                                                                                                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                         |
|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [**Logistic_Regression_1**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/image_recognition1.py)   | A simple logistic regression model for image recognition, implemented from scratch using NumPy. This model is a basic classification model used as an introduction to deep learning concepts.                                                                                                                                                                                                                                       |
| [**Logistic_Regression_2**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/image_recognition2.py)   | A logistic regression model similar to the first but implemented using TensorFlow. This version takes advantage of TensorFlow's functionalities to streamline the model creation and training process for image recognition tasks.                                                                                                                                                                                                    |
| [**FFNN_1**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/classification.py)                      | A shallow feedforward neural network (FFNN) with one hidden layer for Boolean classification tasks. Implemented from scratch using NumPy, this model serves as an introduction to neural networks.                                                                                                                                                                                                                                   |
| [**FFNN_2**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/deep_FNN1.py)                           | A deep feedforward neural network (FFNN), implemented from scratch using NumPy. This model is used for Boolean classification tasks and employs gradient descent as the optimization algorithm. It was trained on a cat dataset to predict whether an image contains a cat.                                                                                                                                                          |
| [**FFNN_3**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/deep_FNN2.py)                           | A deep feedforward neural network (FFNN) similar to FFNN_2, but with multiple parameter initialization options, including He, Xavier, and Gaussian random variables. This model is designed to compare different initialization methods for Boolean classification tasks.                                                                                                                                                             |
| [**FFNN_4**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/L2_DFNN.py)                             | A deep feedforward neural network (FFNN) with L2 regularization, implemented from scratch using NumPy. L2 regularization is employed to prevent overfitting. This model was trained on a synthetic dataset to evaluate the effectiveness of L2 regularization in improving generalization.                                                                                                                                           |
| [**FFNN_5**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/Adam_DFFNN.py)                          | A deep feedforward neural network (FFNN) with Adam optimization, mini-batch gradient descent, and stochastic gradient descent techniques. The model includes options for bias correction and dynamic learning rate adjustment. It was trained on both a synthetic 2D dataset and a cat image dataset, using the Adam optimizer to enhance convergence speed and performance.                                                             |
| [**FFNN_6**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/DFNN_dropout.py)                        | A deep feedforward neural network (FFNN) with dropout regularization, implemented from scratch using NumPy. Dropout is used to prevent overfitting and improve generalization in Boolean classification tasks.                                                                                                                                                                                                                      |
| [**FFNN_7**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/multiclass_classification.py)           | A deep feedforward neural network (FFNN) implemented from scratch using NumPy, with Adam optimization for multiclass classification tasks. The model includes learning rate decay and L2 regularization techniques and was trained on a dataset of hand-sign images to classify the numbers each image represents.                                                                                                               |
| [**CNN_1**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/CNN1.py)                                 | A convolutional neural network (CNN) implemented using TensorFlow. The structure of the model includes layers like Conv2D, MaxPooling2D, and FullyConnected layers. The model can be modified to be larger or smaller, and was trained on a hand-sign image dataset.                                                                                                                                                                 |
| [**CNN_2**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/CNN2.py)                                 | A convolutional neural network (CNN) implemented from scratch using NumPy. The model structure includes Conv2D, MaxPooling2D, and FullyConnected layers. This model is designed to provide a deeper understanding of how CNNs work under the hood by implementing all components manually.                                                                                                                                            |
| [**ResNet_50**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/ResNet.py)                           | An implementation of a very deep convolutional neural network using Residual Networks (ResNet50), based on the paper by K. He et al. (2015). The model is implemented using TensorFlow and is designed to address the vanishing gradient problem in deep networks.                                                                                                                                                                  |
| [**U-Net**](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/u_net.py)                                | An implementation of the U-Net architecture, based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by O. Ronneberger et al. (2015). This model is implemented using TensorFlow and is designed for image segmentation tasks, particularly in the biomedical field.                                                                                                                                  |


This project is licensed under the [MIT LICENSE](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/LICENSE)

