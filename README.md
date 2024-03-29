# Tensorflow-MNIST CNN
Building High Performance Convolutional Neural Networks with TensorFlow

# Model Description
This model is an improvement of the original LeNet-5 model. Features:
* Image Sub-sampling with Convolutional (CNN) Layers
* Reguralization Layers (Batch Normalization, Dropouts)
* Yogi Optimizer (An improved version of Adam optimizer: https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf)
* AdaBelief + LookAhead Optimizer (State of the Art Adam extension optimizer: https://arxiv.org/abs/2010.07468)
* Output noise with Label Smoothing (https://arxiv.org/pdf/2011.12562.pdf)
* Early Stopping Mechanism
* Learning Rate Decay
* Weight Decay
* UMAP - Dimension Reduction

# Model Architecture

![cnn visualization](https://github.com/kochlisGit/Tensorflow-MNIST-State-Of-The-Art/blob/master/model_vis.png)

| Depth | Layer (Type)         | Output Shape         | Param # |
| ----- |-------------------- | -------------------- | ------- |
| 1     | RandomRotation       |  (None, 28, 28, 1)   | 0       |
| 1     | Conv2D(32, 3, 1)     |  (None, 26, 26, 1)   | 288     |
| 1     | BatchNormalization   |  (None, 26, 26, 1)   | 128     |
| 2     | Conv2D(32, 3, 1)     |  (None, 24, 24, 1)   | 9216    |
| 2     | BatchNormalization   |  (None, 24, 24, 1)   | 128     |
| 3     | Conv2D(32, 5, 2)     |  (None, 12, 12, 1)   | 25632   |
| 3     | BatchNormalization   |  (None, 12, 12, 1)   | 0       |
| 4     | Conv2D(64, 3, 1)     |  (None, 10, 10, 64)) | 18432   |
| 4     | BatchNormalization   |  (None, 10, 10, 64)  | 256     |
| 5     | Conv2D(64, 3, 1)     |  (None, 8, 8, 64)    | 36864   |
| 5     | BatchNormalization   |  (None, 8, 8, 64)    | 256     |
| 6     | Conv2D(64, 5, 2)     |  (None, 4, 4, 64)    | 102464  |
| 6     | BatchNormalization   |  (None, 4, 4, 64)    | 0       |
| 7     | Conv2D(128, 3, 1)    |  (None, 2, 2, 128)   | 204928  |
| 7     | Flatten              |  (None, 512)         | 204928  |
| 8     | Dense(10)            |  (None, 10)          | 5130    |
                                                            
---------------------------------------------------------------------------------

# Additional Algorithms:
1. UMAP - KNN
2. Ensemble (CNN + UMAP - KNN)

# Results

| Algorithm                   | Accuracy         |
| ----------------------------|------------------|
| UMAP + KNN                  | 0.9595           |
| CNN (Yogi)                  | 0.9956           |
| CNN (AdaBelief) + LookAhead | 0.9963           |
| Ensemble (CNN + UMAP -KNN)  | 0.9596           |
