import random
import torch
import numpy as np

from collections import deque

class Replay_Buffer:
    def __init__(self,
                 max_lenght = 10000,
                 device = "cpu"
                ):

        self.max_lenght = max_lenght
        self.queue = deque(maxlen= max_lenght)

        self.device = device

    def get_len(self):
        return self.queue.__len__()

    def get_elem(self):
        return self.queue.pop()

    def add_elem(self, elem):
        self.queue.append(elem)

    def get_batch(self, batch_size):
        batch = random.sample(self.queue, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        done = np.array(done, dtype=np.float32)
        #Convert each batch of elem into a tensor and return those batch separated
        return (
            torch.tensor(state).to(self.device),
            torch.tensor(action).to(self.device),
            torch.tensor(reward).to(self.device),
            torch.tensor(next_state).to(self.device),
            torch.tensor(done).to(self.device)
        )