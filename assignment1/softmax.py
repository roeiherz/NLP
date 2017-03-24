import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem assignment1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Implementation Matrix

        # Get the maximum each row from data
        max_rows = np.max(x, axis=1)
        # Reshape the max rows to a matrix [:,assignment1]
        reshape_max_rows = max_rows.reshape((max_rows.shape[0]), 1)
        # Normalize the matrix by subtract the max from each row per row
        norm_data = x - reshape_max_rows
        # Power mat by exponent
        exp_mat = np.exp(norm_data)
        # Sum each col per exp_mat
        exp_mat_rows_sum = np.sum(exp_mat, axis=1)
        # The new SoftMax mat is exp_mat normalized by the rows_sum
        x = exp_mat / exp_mat_rows_sum

    else:
        # Implementation a row Vector

        # Get the maximum each row from data
        max_rows = np.max(x)
        # Normalize the matrix by subtract the max from each row per row
        norm_data = x - max_rows
        # Power mat by exponent
        exp_mat = np.exp(norm_data)
        # Sum each col per exp_mat
        exp_mat_rows_sum = np.sum(exp_mat, axis=0)
        # The new SoftMax mat is exp_mat normalized by the rows_sum
        x = exp_mat / exp_mat_rows_sum
        return x

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    pass
    print "Running your tests..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
