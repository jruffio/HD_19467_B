import numpy as np

# Mostly written by ChatGPT

def generate_samples(mu, cov_matrix, num_samples):
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)

    # Generate independent samples from standard normal distribution
    Z = np.random.normal(size=(2, num_samples))

    # Generate correlated samples
    samples = np.dot(L, Z) + np.array(mu)[:, np.newaxis]

    return samples


# Example covariance matrix
cov_matrix = np.array([[1.0**2, 0.9], [0.9, 1.5**2]])

# Example mean vector
mu = [0.0,0.0]

# Number of samples to generate
num_samples = 100000

# Generate samples
samples = generate_samples(mu, cov_matrix, num_samples)

# Print the generated samples
print("Generated samples:")
print(samples)
print(np.nanstd(samples,axis=1))

import matplotlib.pyplot as plt
plt.scatter(samples[0],samples[1])
plt.show()