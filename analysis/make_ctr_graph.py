import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sys = np.load("../results/ctr_k=2_np=5.npy")

    plt.plot(sys)

    plt.title("Final Cumulative Take Rate {:.2f}".format(sys[-1]))

    plt.legend(["Systematic Resampling CTR"])

    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Take Rate")

    plt.savefig("../report/cumulative_take_rate_k=2_np=5.png", bbox_inches='tight')
