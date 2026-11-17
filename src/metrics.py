import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(metrics, range_of_avrg = 100):

    #A simple function to aggregate the metrics produced during training and
    #testing of the agents. A running average per 100 steps is used to make
    #the plots smother and more readable.

    #Input:
    #-ret : a list of vector each one containing the samples from each
    #       episode relative to a specific metric.
    #Output:
    #-return a list o vectors with the running average of the metrics

    rewards = metrics[0]
    steps = metrics[1]
    success = metrics[2]

    convergence_time = len(rewards)

    mov_avg_suc = np.convolve(success, np.ones(100)/100, mode='valid')
    mov_avg_rew = np.convolve(rewards, np.ones(100)/100, mode='valid')
    mov_avg_steps = np.convolve(steps, np.ones(100)/100, mode='valid')

    for i in range(len(mov_avg_suc)):
        if mov_avg_suc[i] >= 0.9:
          convergence_time = min(convergence_time, i)
        else:
          convergence_time = len(rewards)

    print("\n Number of Total Success: ", success.count(1),
          " Giving a success rate of: ", success.count(1)/len(success),
          "with a total reward of: ", sum(rewards),
          "convergence time: ", convergence_time)

    return [mov_avg_rew, mov_avg_steps, mov_avg_suc]

def show_plot(r, labels):

    #This function as the goal of producing plots for the metrics produced by 2
    #different models, to compare them in a single plot for each of the 3 metric.

    #A specific plot is build for each metric. In each plot are reported the
    #performances of two agents to allow a better comparison between them.
    #The plot are builded taking as input what follows
    #Input:
    #r: set of metrics produced during training and testing by the 2 agents and
    #   used to build the plots.
    #label1: name of the agent that produced the first set of metrics in r
    #label2: name of the agent that produced the second set of metrics in r



    fig, (plot1, plot2, plot3) = plt.subplots(3, 1, figsize=(10, 15))

    for i in range(len(r)):
        label = labels[i]
        
        plot1.plot(r[i][0][0], label= label )
        plot2.plot(r[i][0][1], label= label)
        plot3.plot(r[i][0][2], label= label)

    plot1.set_title("avg reward")
    plot1.set_xlabel("Episode")
    plot1.set_ylabel("avg reward")
    plot1.legend()

    plot2.set_title("average number of steps")
    plot2.set_xlabel("Episode")
    plot2.set_ylabel("steps")
    plot2.legend()

    plot3.set_title("avg success rate")
    plot3.set_xlabel("Episode")
    plot3.set_ylabel("success")
    plot3.legend()

    plt.show()