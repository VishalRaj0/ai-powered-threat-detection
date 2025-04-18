import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the synthetic data
df = pd.read_csv('synthetic_network_data.csv')  # or use df from memory if already created
X = df.drop('label', axis=1)
y = df['label']

# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[y == 0, 0], X_reduced[y == 0, 1], c='green', label='Normal', alpha=0.5)
plt.scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], c='red', label='Anomaly', alpha=0.7)
plt.title('PCA Visualization of Network Traffic')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
