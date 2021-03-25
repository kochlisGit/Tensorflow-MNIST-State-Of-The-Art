from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import umap

# Loading the training data, test data.
(train_inputs, train_targets), (test_inputs, test_targets) = mnist.load_data()
print('Original data:', train_inputs.shape)

# Reshaping training inputs.
rows = train_inputs.shape[0]
height = train_inputs.shape[1]
width = train_inputs.shape[2]

train_inputs = np.reshape(train_inputs, (rows, height*width))
print('Reshaped data:', train_inputs.shape)

# Reducing dimensions to 2.
reducer = umap.UMAP(random_state=42)
low_dim_data = reducer.fit_transform(train_inputs, y=train_targets)
print('Low dimension data data:', low_dim_data.shape)

# Plotting dataset.
_, ax = plt.subplots()
color = train_targets.astype(int)
plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1], c=color, cmap='Spectral', s=0.1)
ax.set_title('MNIST Dataset Visualization', fontsize=16)
plt.show()
