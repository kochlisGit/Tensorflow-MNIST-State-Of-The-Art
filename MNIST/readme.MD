# Convolutional Neural Network for Digit Recognition
This is my implementation of a simple yet powerful CNN to solve the Digit Recognition problem. To train the CNN You will need the following libraries:
1. Tensorflow
1. Keras
1. Numpy
1. Matplotlib

# Architecture
The architecture of my CNN consists of:
- Three Convolutional Layers with Batch Normalization at the output of each layer and a Dropout layer at the 3rd Convolutional layer.
  - Conv2D: (filters=32, kernel_size=3, stride=1, padding=None, 'Xavier' initializer, 'relu' activation, regularizer=None) --> Batch Normalization
  - Conv2D: (filters=32, kernel_size=3, stride=1, padding=None, 'Xavier' initializer, 'relu' activation, regularizer=None) --> Batch Normalization
  - Conv2D: (filters=32, kernel_size=3, stride=1, padding=same, 'Xavier' initializer, 'relu' activation, regularizer=None) --> Batch Normalization --> Dropout
- Another Three Convolutional Layers with Batch Normalization at the output of each layer and a Dropout layer at the 3rd Convolutional layer.
  - Conv2D: (filters=64, kernel_size=5, stride=2, padding=None, 'Xavier' initializer, 'relu' activation, regularizer=None) --> Batch Normalization
  - Conv2D: (filters=64, kernel_size=5, stride=2, padding=None, 'Xavier' initializer, 'relu' activation, regularizer=None) --> Batch Normalization
  - Conv2D: (filters=64, kernel_size=5, stride=2, padding=same, 'Xavier' initializer, 'relu' activation, regularizer=None) --> Batch Normalization --> Dropout
- Final Conv2D: (filters=128, kernel_size=4, stride=1, padding=none, 'Xavier' initializer, 'relu' activation, regularizer=l1)
- Final Dense: (units=10, activation='softmax', 'Xavier' initializer, regularizer=None)

# Results.
Epoch 1/10

938/938 [==============================] - 486s 518ms/step - loss: 2.5568 - accuracy: 0.9327 - val_loss: 1.3332 - val_accuracy: 0.9654

Epoch 2/10

938/938 [==============================] - 638s 680ms/step - loss: 1.1360 - accuracy: 0.9673 - val_loss: 1.1533 - val_accuracy: 0.9834

Epoch 3/10

938/938 [==============================] - 648s 691ms/step - loss: 0.9754 - accuracy: 0.9744 - val_loss: 0.8854 - val_accuracy: 0.9840

Epoch 4/10

938/938 [==============================] - 587s 626ms/step - loss: 0.8727 - accuracy: 0.9777 - val_loss: 0.7965 - val_accuracy: 0.9890

Epoch 5/10

938/938 [==============================] - 512s 546ms/step - loss: 0.7879 - accuracy: 0.9807 - val_loss: 0.7583 - val_accuracy: 0.9906

Epoch 6/10

938/938 [==============================] - 472s 503ms/step - loss: 0.7209 - accuracy: 0.9822 - val_loss: 0.6124 - val_accuracy: 0.9936

Epoch 7/10

938/938 [==============================] - 482s 514ms/step - loss: 0.6659 - accuracy: 0.9839 - val_loss: 0.6854 - val_accuracy: 0.9916

Epoch 8/10

938/938 [==============================] - 467s 498ms/step - loss: 0.6305 - accuracy: 0.9856 - val_loss: 0.6540 - val_accuracy: 0.9890

Epoch 9/10

938/938 [==============================] - 492s 525ms/step - loss: 0.5825 - accuracy: 0.9862 - val_loss: 0.5511 - val_accuracy: 0.9939

Epoch 10/10

938/938 [==============================] - 466s 496ms/step - loss: 0.5568 - accuracy: 0.9872 - val_loss: 0.5081 - val_accuracy: 0.9931

Test Loss: 0.46450069546699524

Test Accuracy: 0.993399977684021

![image](https://github.com/kochlisGit/Deep-Learning/blob/master/MNIST/training_loss.png)

                precision    recall  f1-score   support

           0       1.00      0.99      1.00       980
           1       0.99      1.00      1.00      1135
           2       0.99      0.99      0.99      1032
           3       0.99      1.00      0.99      1010
           4       0.98      1.00      0.99       982
           5       0.99      0.99      0.99       892
           6       0.99      1.00      0.99       958
           7       0.99      0.99      0.99      1028
           8       1.00      0.99      0.99       974
           9       1.00      0.97      0.98      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000

      0     1    2    3    4    5    6    7    8    9

0 [[ 974    0    0    0    0    0    3    2    1    0]
1  [   0 1131    0    1    1    1    1    0    0    0]
2  [   0    0 1022    0    0    0    6    4    0    0]
3  [   0    0    1 1006    0    2    0    1    0    0]
4  [   0    0    0    0  978    0    3    0    0    1]
5  [   0    0    0    7    0  882    1    1    0    1]
6  [   1    0    0    0    1    1  955    0    0    0]
7  [   0    4    3    0    0    0    0 1021    0    0]
8  [   0    0    2    1    1    1    0    1  968    0]
9  [   1    2    0    0   20    1    0    3    3  979]]
