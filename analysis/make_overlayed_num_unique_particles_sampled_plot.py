import numpy as np
import matplotlib.pyplot as plt


def moving_avg(x):
    avgs = []
    for i, v in enumerate(x):
        avgs.append(np.sum(x[:i]) / i)
    return avgs

if __name__ == "__main__":
    sys = np.load("../results/systematic_resampling_unique_particles.npy")
    strat = np.load("../results/stratified_resampling_unique_particles.npy")
    multi = np.load("../results/multinomial_resampling_unique_particles.npy")

    plt.plot(moving_avg(np.array(sys)[:10000]))
    plt.plot(moving_avg(np.array(strat)[:10000]))
    plt.plot(moving_avg(np.array(multi)[:10000]))

    plt.title("Average Number of Unique Particles Sampled")

    plt.legend(["Systematic Resampling", "Stratified Resampling", "Multinomial Resampling"])

    plt.xlabel("Iteration")
    plt.ylabel("Mean Unique Particles")

    plt.savefig("../report/num_unique_particles_when_resampling_k=2_np=10.png", bbox_inches='tight')
