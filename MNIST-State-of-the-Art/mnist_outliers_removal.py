from keras.datasets import mnist
from sklearn.cluster import DBSCAN
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
data_clusters = DBSCAN(n_jobs=-1).fit_predict(mapper.embedding_)

# Plotting outlying digits.
outlying_digits = train_inputs[data_clusters == -1]
print('Number of outlyings digits found:', outlying_digits.shape[0])

fig, axes = plt.subplots(4, 8, figsize=(10,10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(outlying_digits[i].reshape((28,28)))
plt.tight_layout()
plt.title('Outlying digits')
plt.show()


