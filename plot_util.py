from matplotlib import pyplot as plt
import numpy as np


def plot_progress(values, exploration_rate, average_period=100):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(values, label="Reward in episode x")
    plt.plot(get_average(values, average_period), label="Average reward per " + str(average_period) + " episodes")
    plt.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.2)
    # plt.gcf().text(0.02, -0.1, "Exploration rate: " + str(exploration_rate), fontsize=12)
    # plt.annotate("Test", [0, -20])
    plt.text(0.02, 0.025, "Exploration rate: %.2f" % exploration_rate, fontsize=10, transform=plt.gcf().transFigure)
    # plt.autoscale()
    # plt.show()
    plt.pause(0.0001)


def get_average(values, period):
    if len(values) >= period:
        cumsum = np.cumsum(np.insert(values, 0, 0))
        return np.concatenate((np.zeros(period), (cumsum[period:] - cumsum[:-period]) / float(period)))
    else:
        return np.zeros(len(values))


# def get_average_manual(values, period):
#     avg = []
#     for i in range(len(values)):
#         if i < period:
#             avg.append(0)
#         else:
#             sum = np.sum(values[i - period:i])
#             avg.append(sum / float(period))
#     return avg

