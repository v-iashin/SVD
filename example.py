# make sure to import numpy
import numpy as np

# the function SVD takes parameters such as:
# learning rate
alpha = 0.001
# number of iterations
n = 1000
# the n-by-m matrix that we want to approximate
A = np.array([[3, 2], [-4, 6], [-1, 3]])
# and a couple of vectors: b is n by k
B = np.array([[1], [4], [2]])
# and c is m by k
C = np.array([[-3], [2]])

# run
SVD(mat = A, initial_vec1 = B, initial_vec2 = C, learn_rate = alpha, iterations = n)
