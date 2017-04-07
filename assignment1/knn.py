import numpy as np
import scipy

def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    #normzlize matrix, vectors and sort according to dot product
    norm_matrix = matrix / np.sqrt((matrix * matrix).sum(axis=1)).reshape(-1, 1)
    norm_vec = vector / np.sqrt((vector * vector).sum(axis=0))
    cosine_dist_vec = np.abs(np.dot(norm_matrix, norm_vec.reshape(-1, 1)))
    nearest_idx = cosine_dist_vec.argsort(axis=0)[-k:][::-1]
    return nearest_idx.reshape(-1)

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    test_knn()


