
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


def p_x(x, mu, sigma):
    z1 = 0.3
    z2 = 1 - z1
    g1 = stats.norm(mu[0], sigma[0]).pdf(x)
    g2 = stats.norm(mu[1], sigma[1]).pdf(x)
    y = z1 * g1 + z2 * g2
    return y


def q_x(x, mu, sigma):
    """
        use Gaussian distribution as upper bound distribution
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    left_bound = mu[0] - 3 * sigma[0]
    right_bound = mu[1] + 3 * sigma[1]
    mu = (right_bound + left_bound) / 2
    sigma = (right_bound - left_bound) / 6

    g = stats.norm(mu, sigma).pdf(x)
    return g


def q_x_uniform(x, mu, sigma):
    """
        use uniform distribution as upper bound distribution
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    left_bound = mu[0] - 3 * sigma[0]
    right_bound = mu[1] + 3 * sigma[1]
    u = stats.uniform(left_bound, right_bound).pdf(x)
    return u


def reject_sample(n_iterations, mus, sigmas):
    left_bound = mus[0] - 3 * sigmas[0]
    right_bound = mus[1] + 3 * sigmas[1]
    mu_upper = (right_bound + left_bound) / 2
    sigma_upper = (right_bound - left_bound) / 6
    x_list = []

    for i in range(n_iterations):
        x_i = np.random.normal(mu_upper, sigma_upper)
        y_i = q_x(x_i, mus, sigmas)
        # y_i = q_x_uniform(x_i, mus, sigmas)
        z = np.random.uniform(0, y_i)
        if z < p_x(x_i, mus, sigmas):
            x_list.append(x_i)

    return x_list


if __name__ == '__main__':
    mus = [10, 25]
    sigmas = [3, 5]
    x = np.arange(1, 40, 1)

    y_list = p_x(x, mus, sigmas)
    plt.plot(x, y_list, c='red')

    # y_upper = q_x(x, mus, sigmas)
    y_upper = q_x_uniform(x, mus, sigmas)
    M = max(y_list/y_upper)
    plt.plot(x, M*y_upper, c='blue')

    x_sample = reject_sample(50000, mus, sigmas)
    plt.hist(x_sample, normed=True, bins=100, facecolor="green",
             edgecolor="black", alpha=0.7)

    plt.show()