import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1347)

# goal: estimate pi by using unit-circle

def uniform_dist():
    x = np.random.random_sample()
    y = np.random.random_sample()
    return x, y


def pi_estimation(n_samples):
    circle_count = 0
    circle_point = [[], []]
    outer_point = [[], []]
    for i in range(n_samples):
        # use uniform to generate samples
        x, y = uniform_dist()
        if x**2 + y**2 <= 1:
            circle_count += 1
            circle_point[0].append(x)
            circle_point[1].append(y)
        else:
            outer_point[0].append(x)
            outer_point[1].append(y)

    pi = (circle_count/n_samples)*4
    return pi, circle_point, outer_point


if __name__ == '__main__':
    samples = [50, 100, 500, 1000, 2000, 5000, 10000, 15000,
               20000, 25000, 30000, 35000, 40000, 45000, 50000]
    sample = [samples[np.random.randint(0, len(samples))]]

    pi = []
    fig = plt.figure()

    # samples is bigger gradually, pi is steady about 3.14
    for i, item in enumerate(sample):
        print(item)
        pi_e, circle_point, outer_point = pi_estimation(item)
        ax1 = plt.subplot(1, len(sample), i+1)
        ax1.scatter(circle_point[0], circle_point[1], c='red', alpha=0.5)
        ax1.scatter(outer_point[0], outer_point[1], c='blue', alpha=0.5)
        plt.show()

        pi.append(pi_e)
        print(pi_e)

    # ax2 = plt.subplot(1, len(samples)+1, len(samples)+1)
    # ax2.scatter(samples, pi)
    # plt.show()
