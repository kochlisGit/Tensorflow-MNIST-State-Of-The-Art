# Tensorflow-State-of-the-Art-Neural-Networks
Building High Performance Convolutional Neural Networks with TensorFlow

# Model Description
This model is an improvement of the original LeNet-5 model. Features:
* Image Sub-sampling with Convolutional (CNN) Layers
* Reguralization Layers (Batch Normalization, Dropouts)
* Yogi Optimizer (An improved version of Adam optimizer: https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf)
* Output noise with Label Smoothing (https://arxiv.org/pdf/2011.12562.pdf)

# Model Architecture

![cnn visualization](https://github.com/kochlisGit/Tensorflow-MNIST-State-Of-The-Art/blob/master/model_vis.png)

| Layer (Type)         | Output Shape         | Param # |
| -------------------- | -------------------- | ------- |
| RandomRotation       |  (None, 28, 28, 1)   | 0       |
| Conv2D(32, 3, 1)     |  (None, 26, 26, 1)   | 288     |
| BatchNormalization   |  (None, 26, 26, 1)   | 128     |
| Conv2D(32, 3, 1)     |  (None, 24, 24, 1)   | 9216    |
| BatchNormalization   |  (None, 24, 24, 1)   | 128     |
| Conv2D(32, 5, 2)     |  (None, 12, 12, 1)   | 25632   |
| BatchNormalization   |  (None, 12, 12, 1)   | 0       |
| Conv2D(64, 3, 1)     |  (None, 10, 10, 64)) | 18432   |
| BatchNormalization   |  (None, 10, 10, 64)  | 256     |
| Conv2D(64, 3, 1)     |  (None, 8, 8, 64)    | 36864   |
| BatchNormalization   |  (None, 8, 8, 64)    | 256     |
| Conv2D(64, 5, 2)     |  (None, 4, 4, 64)    | 102464  |
| BatchNormalization   |  (None, 4, 4, 64)    | 0       |
| Conv2D(128, 3, 1)    |  (None, 2, 2, 128)   | 204928  |
| Flatten              |  (None, 512)         | 204928  |
| Dense(10)            |  (None, 10)          | 5130    |

 Layer (type)                Output Shape              Param #   
=================================================================
 random_rotation_1 (RandomRo  (None, 28, 28, 1)        0         
 tation)                                                         
                                                                 
 conv2d_21 (Conv2D)          (None, 26, 26, 32)        288       
                                                                 
 batch_normalization_12 (Bat  (None, 26, 26, 32)       128       
 chNormalization)                                                
                                                                 
 conv2d_22 (Conv2D)          (None, 24, 24, 32)        9216      
                                                                 
 batch_normalization_13 (Bat  (None, 24, 24, 32)       128       
 chNormalization)                                                
                                                                 
 conv2d_23 (Conv2D)          (None, 12, 12, 32)        25632     
                                                                 
 dropout_6 (Dropout)         (None, 12, 12, 32)        0         
                                                                 
 conv2d_24 (Conv2D)          (None, 10, 10, 64)        18432     
                                                                 
 batch_normalization_14 (Bat  (None, 10, 10, 64)       256       
 chNormalization)                                                
                                                                 
 conv2d_25 (Conv2D)          (None, 8, 8, 64)          36864     
                                                                 
 batch_normalization_15 (Bat  (None, 8, 8, 64)         256       
 chNormalization)                                                
                                                                 
 conv2d_26 (Conv2D)          (None, 4, 4, 64)          102464    
                                                                 
 dropout_7 (Dropout)         (None, 4, 4, 64)          0         
                                                                 
 conv2d_27 (Conv2D)          (None, 2, 2, 128)         204928    
                                                                 
 flatten_3 (Flatten)         (None, 512)               0         
                                                                 
 dense_3 (Dense)             (None, 10)                5130      
                                                                 
=================================================================
Total params: 403,722
Trainable params: 403,338
Non-trainable params: 384
_________________________________________________________________
