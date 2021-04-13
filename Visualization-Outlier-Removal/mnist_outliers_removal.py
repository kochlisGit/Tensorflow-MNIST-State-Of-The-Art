from keras.datasets import mnist
import hdbscan
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

mapper = umap.UMAP(set_op_mix_ratio=0.25, random_state=42).fit(train_inputs, y=train_targets)
data_clusters = hdbscan.HDBSCAN().fit_predict(mapper.embedding_)

# Plotting outlying digits.
outlying_digits = train_inputs[data_clusters == -1]
print('Number of outlyings digits found:', outlying_digits.shape[0])

fig, axes = plt.subplots(5, 6, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(outlying_digits[i].reshape((28, 28)))
plt.tight_layout()
plt.title('Outlying digits')
plt.show()

# Plotting normal digits vs outlier digits.
clear_digits = mapper.embedding_[data_clusters != -1]
outlying_digits = mapper.embedding_[data_clusters == -1]

# Plotting dataset.
_, ax = plt.subplots()
color = train_targets.astype(int)
plt.scatter(clear_digits[:, 0], clear_digits[:, 1], c='blue', cmap='Spectral', s=0.1)
plt.scatter(outlying_digits[:, 0], outlying_digits[:, 1], c='red', cmap='Spectral', s=0.1)
ax.set_title('MNIST Dataset Visualization', fontsize=16)
plt.show()
