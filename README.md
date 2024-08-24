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



| Model                                                                                                 | Description                                                                                                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ***[FFNN_1](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/deep_FNN1.py)***  | A deep feedforward neural network (MLP) implemented from scratch using Numpy. This model is used for Boolean classification tasks and employs gradient descent as the optimization algorithm. It was trained on a cat dataset to predict whether an image is of a cat or not.                                                                                     |
| ***[FFNN_2](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/deep_FNN2.py)***  | A deep feedforward neural network (MLP), similar to FFNN_1 but with multiple parameter initialization choices. It includes options like ***He***, ***Xavier***, and ***Gaussian random variables*** for weight initialization. This variation aims to compare different initialization methods on Boolean classification tasks.                                 |
| ***[FFNN_3](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/L2_DFNN.py)***    | A deep feedforward neural network (MLP) with L2 regularization implemented from scratch using Numpy. L2 regularization is used to prevent overfitting. This model was trained on a synthetic dataset to evaluate the effectiveness of L2 regularization in improving generalization.                                                                                           |
| ***[FFNN_4](https://github.com/LoqmanSamani/deep-learning/blob/systembiology/models/Adam_DFFNN.py)*** | A deep feedforward neural network (MLP) with Adam optimization and mini-batch/stochastic gradient descent techniques. The model includes options for bias correction and dynamic learning rate adjustment. It was trained on both a synthetic 2D dataset and a cat image dataset. The Adam optimizer is used to improve convergence speed and performance. |
| ***[]()***                                                                                            ||









This project is licensed under the [MIT LICENSE](https://github.com/LoqmanSamani/machine-learning/blob/systembiology/LICENSE)

