def SVD(mat, initial_vec1, initial_vec2, learn_rate, iterations):
    ## reassigning vectors in order to keep code clean
    # the matrix that we want to approximate
    A = mat
    # two vectors from which we will start
    b = initial_vec1
    c = initial_vec2
    # learning rate, or step of learning
    alpha = learn_rate
    # number of iterations
    n = iterations
    # A ~ b * c^t : the first approximation based on given initial vectors
    A_app = np.dot(b, c.T)
    # loop
    for i in range(n):
        # partial derivatives for vectors
        dLdb = np.dot((A_app - A), c)
        dLdc = np.dot((A_app - A).T, b)
        # updating vectors
        c = c - alpha * dLdc
        b = b - alpha * dLdb
        # calculating approximated matrix
        A_app = np.dot(b, c.T)
    # returning two vectors that can be used for A approximation
    return b, c, A_app
