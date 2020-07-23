import tensorflow as tf
import spambase_dataset

# Importing data and preprocessing data.
spambase_dataset.import_data()
spambase_dataset.preprocess(standarization=True)

# Defining NN Architecture.
tf.reset_default_graph()

input_size = spambase_dataset.input_size
output_size = 2
hidden_layer_size1 = 20
hidden_layer_size2 = 10

# Feeding data to placeholders.
inputs = tf.placeholder( tf.float32, [None, input_size] )
targets = tf.placeholder( tf.float32, [None, output_size] )

# Defining weights of 1st layer, with initializer and regularizer.
weights_1 = tf.get_variable(name='weights_1', shape=[input_size, hidden_layer_size1],
                            initializer=tf.initializers.glorot_normal,
                            regularizer=tf.keras.regularizers.l2(0.01) )
biases_1 = tf.get_variable(name='biases_1', shape=[hidden_layer_size1],
                            initializer=tf.initializers.glorot_normal,
                            regularizer=tf.keras.regularizers.l2(0.01) )

# Defining activation function of 1st layer.
layer_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

# Defining output of 1st layer.
outputs_1 = tf.nn.dropout(layer_1, rate=0.1)

weights_2 = tf.get_variable(name='weights_2', shape=[hidden_layer_size1, hidden_layer_size2],
                            initializer=tf.initializers.glorot_normal,
                            regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) )
biases_2 = tf.get_variable(name='biases_2', shape=[hidden_layer_size2],
                            initializer=tf.initializers.glorot_normal,
                            regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) )
layer_2 = tf.nn.elu(tf.matmul(outputs_1, weights_2) + biases_2)
outputs_2 = tf.nn.dropout(layer_2, rate=0.1)

weights_3 = tf.get_variable(name='weights_3', shape=[hidden_layer_size2, output_size],
                            initializer=tf.initializers.glorot_normal,
                            regularizer=tf.keras.regularizers.l2(0.01) )
biases_3 = tf.get_variable(name='biases_3', shape=[output_size],
                            initializer=tf.initializers.glorot_normal,
                            regularizer=tf.keras.regularizers.l2(0.01) )

outputs = tf.matmul(outputs_2, weights_3) + biases_3

# Selecting loss function.
loss_func = tf.losses.huber_loss(labels=targets, predictions=outputs)
mean_loss = tf.reduce_mean(loss_func)

# Selecting optimization algorithm.
optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=0.005).minimize(loss=mean_loss)

# Compare predictions with targets.
out_equals_target = tf.equal( tf.argmax(outputs, 1), tf.argmax(targets, 1) )
accuracy = tf.reduce_mean( tf.cast(out_equals_target, tf.float32) )

# Initializing weights.
sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()
sess.run(initializer)

# Defining epochs and early stopping method.
min_epochs = 50
max_epochs = 500
e = 0
s = 10
E_val_strips = [0]*s

print('Training model...\n')

while(e < max_epochs):
    print('\nepoch =', e)

    E_tr = 0
    E_val = 0
    E_opt = 999999.
    acc = 0
    
    for train_inputs, train_targets, valid_inputs, valid_targets in spambase_dataset.next_batch():
        _, batch_train_loss = sess.run( [optimizer, mean_loss],
                                    feed_dict={inputs: train_inputs, targets: train_targets} )

        E_tr += batch_train_loss

        validation_loss, validation_accuracy = sess.run( [mean_loss, accuracy],
                                                            feed_dict={inputs: valid_inputs, targets: valid_targets} )
        E_val += validation_loss
        acc += validation_accuracy

        if validation_loss < E_opt:
            E_opt = validation_loss
    
    E_tr /= spambase_dataset.k
    E_val /= spambase_dataset.k
    acc /= spambase_dataset.k

    print('Training loss =', E_tr)
    print('Validation loss =', E_val)
    print('Accuracy =', acc)
    
    if e < min_epochs:
        E_val_strips[e % s] = E_val
    else:
        if E_val > max(E_val_strips):
            print('Training stopped as Validation Error is larger than the last', s, 'epochs.')
            break
        else:
            E_val_strips[e % s] = E_val
    e += 1
    
print('\nEnd of training.')

# Testing the model...
test_inputs, test_targets = spambase_dataset.test_batch()
test_accuracy = sess.run( [accuracy],
                            feed_dict={inputs: test_inputs, targets: test_targets} )
test_acc_percentage = test_accuracy[0] * 100

print('Test accuracy =', test_acc_percentage)