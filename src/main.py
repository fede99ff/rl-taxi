import torch

import metrics as met
import tabular_agents as tab
import DQN_agents as dqn

import random
import numpy as np
import gymnasium as gym


def set_seed(seed=42):
    #Here all the libraries using some random process have their seed
    #set for reproducibility porpuses.
    random.seed(seed)                 # Python's random module
    np.random.seed(seed)              # NumPy
    torch.manual_seed(seed)           # PyTorch (CPU)
    torch.cuda.manual_seed(seed)      # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    gym.utils.seeding.np_random(seed) # Gym environment

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Here tabular agents are trained and tested. The metrics for both are collected,
#elaborated and stored to be plotted later

train_results = []
test_results = []

#print("\n --------------------------------------------------------STARTING THE TRAINING OF THE single QL AGENT-------------------------------------------------------- \n")
#train_results.append([met.compute_metrics(tab.run_QL(6000, "QL", 0.1, 0.9, 0.9994,  True, False))])
#print("\n --------------------------------------------------------STARTING THE TRAINING OF THE single SARSA AGENT-------------------------------------------------------- \n")
#train_results.append([met.compute_metrics(tab.run_QL(6000, "SARSA", 0.1, 0.9, 0.9994,  True, False))])
#
#print("\n --------------------------------------------------------STARTING THE TESTING OF THE single QL AGENT-------------------------------------------------------- \n")
#test_results.append([met.compute_metrics(tab.run_QL(50000, "QL"), 5000)])
#print("\n --------------------------------------------------------STARTING THE TESTING OF THE single SARSA AGENT-------------------------------------------------------- \n")
#test_results.append([met.compute_metrics(tab.run_QL(50000, "SARSA"), 5000)])
#
#met.show_plot(train_results, ["QL", "SARSA"])
#met.show_plot(test_results, ["QL", "SARSA"])
#
#
##Here DQN agents are trained and tested. The metrics for both are collected,
##elaborated and stored to be plotted later
#
#train_results = []
#test_results = []
#
#print("\n --------------------------------------------------------STARTING THE TRAINING OF THE single DQN AGENT-------------------------------------------------------- \n")
#train_results.append([met.compute_metrics(dqn.run_single_DQL(30000, "DQN", device, 0.001, 0.999, 0.00005,  True, False))])
#print("\n --------------------------------------------------------STARTING THE TRAINING OF THE TARGET DQN AGENT-------------------------------------------------------- \n")
#train_results.append([met.compute_metrics(dqn.run_single_DQL(30000, "TargetDQN", device, 0.001, 0.999, 0.00005,  True, False))])
#print("\n --------------------------------------------------------STARTING THE TRAINING OF THE DDQN AGENT-------------------------------------------------------- \n")
#train_results.append([met.compute_metrics(dqn.run_single_DQL(30000, "DDQN", device, 0.001, 0.999, 0.00005,  True, False))])

print("\n --------------------------------------------------------STARTING THE TESTING OF THE single DQN AGENT-------------------------------------------------------- \n")
test_results.append([met.compute_metrics(dqn.run_single_DQL(50000, "DQN", device))])
print("\n --------------------------------------------------------STARTING THE TESTING OF THE TARGET DQN AGENT-------------------------------------------------------- \n")
test_results.append([met.compute_metrics(dqn.run_single_DQL(50000, "TargetDQN", device))])
print("\n --------------------------------------------------------STARTING THE TESTING OF THE DDQN AGENT-------------------------------------------------------- \n")
test_results.append([met.compute_metrics(dqn.run_single_DQL(50000, "DDQN", device))])

met.show_plot(train_results, ["Single DQN", "DQN + target network", "DDQN"])
met.show_plot(test_results, ["Single DQN", "DQN + target network", "DDQN"])