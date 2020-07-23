# Spambase

Spambase is my first trained MLP. It's goal is to predict whether an email is spam or not.

# Dataset

You can find the dataset that i used to train the model here: http://archive.ics.uci.edu/ml/datasets/Spambase/

The dataset contains 4600 observations. According to the UCI website, each observation consists of:

* 48 continuous real [0,100] attributes of type word_freq_WORD
= percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.

* 6 continuous real [0,100] attributes of type char_freq_CHAR]
= percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail

* 1 continuous real [1,...] attribute of type capital_run_length_average
= average length of uninterrupted sequences of capital letters

* 1 continuous integer [1,...] attribute of type capital_run_length_longest
= length of longest uninterrupted sequence of capital letters

* 1 continuous integer [1,...] attribute of type capital_run_length_total
= sum of length of uninterrupted sequences of capital letters
= total number of capital letters in the e-mail

* 1 nominal {0,1} class attribute of type spam
= denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.

To test the accuracy of my model, i first splitted the data as:

*1/10 of the data was randomly selected as "Test Input".

*The rest 9/10 was used as "Training Input"

# Preprocessing

I tried different methods of preprocessing, such as Normalization, Standarization, pca with Whitening and even Kernel Pca.
Interestingly, pca and kernel pca did great job at increasing validation accuracy (99+ %), however, the test accuracy was very low (50%). This happened because the model overfitted the training input, even though I used many methods to prevent overfitting. The winner was Standarization, as it performed slightly better accuracy results than normalization method.

# NN Architecture
I used 2 hidden layers.

* The 1st hidden layer was made of 20 neurons. 

* The 2nd hidden layer was made of 10 neurons.

# Activation Functions

* 1st Hidden Layer: _relu_

* 2nd Hidden Layer: _elu_

# Loss Function

I used _huber loss_ as my loss function. The main reason is that i selected this particular loss function is that huber loss is not sensitive to outliers.

# Regularization

I used 3 methods of regularization to prevent overfitting:

1. _Dropouts_ (0.1 at the 1st hidden layer and 0.1 at the second hidden layer).

1. _l1_l2 Regularization_ at the 1st hidden layer and _l1 Regularization_ at the second hidden layer. I tried a lot of combinations and this seemed to fit best.

1. _Early Stopping_ if the validation error is larger than the last 10 epochs. Early stopping is triggered after the MLP is trained in at least 50 epochs.

# Initialization

I used the xavier's initialization method as it is also recommended by TensorFlow's team.

# Optimizer

I was between adam and nadam algorithm, but _Nadam's algorithm_ perfomed better in the average testing, with starting learning rate = 0.005.

# Cross Validation

I used the _10-fold cross validation algorithm_ as an additional method to prevent overfitting.

# Metrics & Results

I used the mean accuracy method as my metric method. The results were 94+ % mean accuracy.
