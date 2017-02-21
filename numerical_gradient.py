import numpy as np
from numpy.linalg import norm


def calc(f, x):
    '''
    Calculates the gradient of f(x) numerically.
    We need to perturb each x_i at the time to figure out the gradient for each x_i.
    :param f: forward function
    :param x: an input vector x
    :return: gradient vector
    '''
    eps = 1e-4
    numerical_grad_vector = []

    for i, x_i in enumerate(x):
        # Just perturbate one x_i at the time:
        x[i] = x_i + eps
        f_x_plus_eps = f(x)
        x[i] = x_i - eps
        f_x_minus_eps = f(x)

        # Put the original value back
        x[i] = x_i

        numerical_grad_i = np.divide((f_x_plus_eps - f_x_minus_eps), (2.*eps))

        numerical_grad_vector.append(numerical_grad_i)

    return np.array(numerical_grad_vector).T


def are_similar(deriv_grad, num_grad):
    '''
    :return: True if num_grad and deriv_grad are identical or almost-identical vectors
    '''
    # diff = norm(deriv_grad-numerical_grad) / norm(deriv_grad+numerical_grad)

    assert deriv_grad is not None
    assert num_grad is not None
    assert np.array_equal(num_grad.shape, deriv_grad.shape), \
        "Numerical and Derivated gradients dimen" \
        "sions don't match:\n" + \
        "num_grad: \n%s\n" % num_grad + \
        "deriv_grad: \n%s\n" % deriv_grad

    bools = (deriv_grad - num_grad) < 1e-8
    similar = np.sum(bools) == bools.size

    if not similar:
        raise Exception("Numerical and Derivated gradients are not similar.\n"+
                        "deriv_grad: \n%s\n num_grad: \n%s\n" % (deriv_grad, num_grad))

    return similar


def check(deriv_grad, f, x):
    num_grad = calc(f, x)
    return are_similar(deriv_grad, num_grad)



