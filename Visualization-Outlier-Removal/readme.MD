# MNIST Data Visualization & Outlier removal.
This is a state-of-the-art visualization of MNIST Dataset, using UMAP Algorithm: https://arxiv.org/abs/1802.03426

# Naive Visualization Approach
In the naive visualization approach, I simply fitted the data into the mapper, reduce the components of the images to 2 and visualize the data.

![Naive Visualization Approach](https://github.com/kochlisGit/Deep-Learning/blob/master/Visualization-Outlier-Removal/Plots/visualization.png)

You can immediately notice a lot of "bad" handwritten digits look very similar to other digits. I was able to detect and remove those outliers using UMAP + DBSCAN method.
In the below image, You can see some of the detected outlying digits.

![Outlying digits](https://github.com/kochlisGit/Deep-Learning/blob/master/Visualization-Outlier-Removal/Plots/outlier_removal.png)

You can notice that some of the digits are hard to recognise even for us.

# Visualization with DenseMAP
If You are going to feed the data into a Neural Network, It is recommended that You use densMAP attribute feature, which better preserves Local Density:
https://umap-learn.readthedocs.io/en/latest/densmap_demo.html

![Local Density](https://github.com/kochlisGit/Deep-Learning/blob/master/Visualization-Outlier-Removal/Plots/visualization_densMAP.png)

Finally, UMAP library allows you to feed the **Targets** into the mapper, which acts as an additional information, in order to better separate the digits from the 2D Space.
This method is also called Supervised Dimensionality Reduction.

![Supervised Visualization](https://github.com/kochlisGit/Deep-Learning/blob/master/Visualization-Outlier-Removal/Plots/visualization_supervised.png)

You can see that there are many outlying digits, that however, are easy to detect. Some of the recommended algorithms for outlier removal can be found here:
https://github.com/kochlisGit/ML-Statistics/tree/main/anomaly-detection

Some recommended algorithms for anomaly detection include:
1. Isolation Forest
2. Local Outlier Factor
3. Elliptic Envelope
4. K - Nearest Neighbors
5. K - Means
6. DBSCAN
7. HDBSCAN

It is recommended that You try the above methods to build a state of the art classifier for the MNIST Dataset.
A simple DNN is likely to perform almost excellent. However, a lot of memory and a good GPU that supports CUDA are 
required in order to use UMAP algorithm to both train and test data.
