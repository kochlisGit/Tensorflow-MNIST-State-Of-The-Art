# State of the Art MNIST Classifier

This is an implementation of a similar-like LeNet CNN for MNIST Digit Classification. I have previously implemented that CNN, with a few changes in the parameters,
that it is performing even better than the original LeNet (https://github.com/kochlisGit/Deep-Learning/tree/master/MNIST). However, this time, I have added some
state-of-the-art algorithms using **Tensorflow Addons** library to increase the accuracy even more.

# Optimizer

The first thing I did was to change the optimizer algorithm. In my previous implementation, I choosed **Adam**, because It is robust and widely used in CNNs. However, I discovered that Google's **Yogi** outperforms Adam and even Rectified Adam. More info about Yogi can be found here: https://papers.nips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf

# Activations

Originally, I used the classical **relu** as activation function for all layers, except the outout layer. However, relu is known to have a lot of drawbacks. **Gelu** however, is
a new Activation function that seems to outperform relu, because it keeps all the advantages of relu, without having its drawbacks. The implementation of gelu can be found here: https://arxiv.org/abs/1606.08415

# Other Losses & Layers that were tried

I tried a lot of brand new algorithms that are offered in Tensorflow Addons library, which speeded up the training a little bit. However, none of them seemed to improve the final accuracy of the model.

1. **Maxout** Layer instead of Dropout. 
1. **Lifted Structure** loss instead of Catigorical Cross entropy.
1. **Sparsemax with Sparsemax Loss** instead of Softmax with Catigorical Cross entropy.

# Final Validation

The validation accuracy of this brand new classifier reaches 99.4% accuracy with only 330,000 parameters and without much tuning. Surely, there are other CNNs that perform better than that, however, they use millions of parameters and the training takes too much time. With this CNN however, You can perform as well as other state-of-the-art CNNs, with very little training.
