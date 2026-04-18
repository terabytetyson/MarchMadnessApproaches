import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
v11 = pd.read_csv('submission_v11.csv')
v12 = pd.read_csv('submission_v12.csv')

# Initial Head inspection
print("V11 Head:")
print(v11.head())
print("\nV12 Head:")
print(v12.head())

# Check shape and ID alignment
print(f"\nV11 Shape: {v11.shape}")
print(f"V12 Shape: {v12.shape}")

# Ensure they are sorted by ID to match correctly if needed ---> or join on ID
df = v11.merge(v12, on='ID', suffixes=('_v11', '_v12'))
print(f"Merged Shape: {df.shape}")
print(df.head())

# Extract vectors
vec_v11 = df['Pred_v11'].values.reshape(1, -1)
vec_v12 = df['Pred_v12'].values.reshape(1, -1)

# Calculate Cosine Similarity
cos_sim = cosine_similarity(vec_v11, vec_v12)[0][0]
print(f"Cosine Similarity: {cos_sim}")

# Plotting
plt.figure(figsize=(12, 6))

# Subplot 1: Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(df['Pred_v11'], df['Pred_v12'], alpha=0.3, s=10)
plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Identity line
plt.xlabel('RoBERTa (v11) Predictions')
plt.ylabel('DistilBERT (v12) Predictions')
plt.title(f'Prediction Comparison\nCosine Similarity: {cos_sim:.4f}')
plt.grid(True, alpha=0.3)

# Subplot 2: Difference Distribution
plt.subplot(1, 2, 2)
differences = df['Pred_v11'] - df['Pred_v12']
plt.hist(differences, bins=50, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Difference (v11 - v12)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Differences')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_analysis.png')

# Summary stats
print("\nDifference Stats:")
print(differences.describe())