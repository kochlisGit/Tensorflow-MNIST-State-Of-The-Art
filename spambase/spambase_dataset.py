import csv
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition

train_filepath = 'spambase_train.data'
test_filepath = 'spambase_test.data'

train_inputs = []
train_targets = []
test_inputs = []
test_targets = []
input_size = 0
num_of_train_examples = 0
num_of_test_examples = 0
batch_size = 0
k = 10

def _read_data(filepath):
    with open(filepath) as f:
        reader = csv.reader(f)
        for entry in reader:
            yield entry

def import_data():
    global train_inputs, train_targets, test_inputs, test_targets, num_of_train_examples, num_of_test_examples, input_size, batch_size

    for entry in _read_data(train_filepath):
        train_inputs.append( [ float(x) for x in entry[ 0:len(entry)-1 ] ] )
        target = int( entry[-1] )
        if target == 0:
            train_targets.append( [0,1] )
        else:
            train_targets.append( [1,0] )

    for entry in _read_data(test_filepath):
        test_inputs.append( [ float(x) for x in entry[ 0:len(entry)-1 ] ] )
        target = int( entry[-1] )
        if target == 0:
            test_targets.append( [0,1] )
        else:
            test_targets.append( [1,0] )

    train_inputs = np.array(train_inputs)
    train_targets = np.array(train_targets)
    test_inputs = np.array(test_inputs)
    test_targets = np.array(test_targets)

    num_of_train_examples = len(train_inputs)
    num_of_test_examples = len(test_inputs)
    input_size = len( train_inputs[0] )
    batch_size = int(num_of_train_examples / k)

def preprocess(normalization=False, standarization=False, pca=False, whitening=False, kpca=False):
    global train_inputs, test_inputs, input_size

    if normalization:
        train_inputs = preprocessing.normalize(train_inputs)
        test_inputs = preprocessing.normalize(test_inputs)
    if standarization:
        train_inputs = preprocessing.StandardScaler().fit_transform(train_inputs)
        test_inputs = preprocessing.StandardScaler().fit_transform(test_inputs)
    if pca:
        train_inputs = decomposition.PCA(n_components='mle', whiten=whitening, svd_solver='full').fit_transform(train_inputs)
        test_inputs = decomposition.PCA(n_components='mle', whiten=whitening, svd_solver='full').fit_transform(test_inputs)
    if kpca:
        train_inputs = decomposition.KernelPCA(kernel='rbf', remove_zero_eig=True).fit_transform(train_inputs)
        test_inputs = decomposition.KernelPCA(kernel='rbf', remove_zero_eig=True).fit_transform(test_inputs)
    input_size = len( train_inputs[0] )

def next_batch():
    train_input_batch = []
    train_target_batch = []
    validation_input_batch = []
    validation_target_batch = []

    for batch_counter in range(k):
        v = batch_counter * batch_size
        train_input_batch = np.concatenate( ( train_inputs[0:v], train_inputs[v+batch_size:num_of_train_examples] ) )
        train_target_batch = np.concatenate( ( train_targets[0:v], train_targets[v+batch_size:num_of_train_examples] ) )
        validation_input_batch = train_inputs[v:v+batch_size]
        validation_target_batch = train_targets[v:v+batch_size]
        yield train_input_batch, train_target_batch, validation_input_batch, validation_target_batch

def test_batch():
    return test_inputs, test_targets