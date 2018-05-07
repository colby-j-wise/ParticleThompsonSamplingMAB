import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sys = np.load("../results/k=2_np=5_systematic_mse=1.68.npy")
    strat = np.load("../results/k=2_np=5_strat_mse=1.92.npy")
    multi = np.load("../results/k=2_np=5_multi_mse=2.08.npy")

    plt.plot(multi)
    plt.plot(strat)
    plt.plot(sys)

    plt.title("Mean Squared Error Across Different Resampling Methods")

    plt.legend(["Multinomial Resampling", "Stratified Resampling", "Systematic Resampling"])

    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")

    plt.show()


