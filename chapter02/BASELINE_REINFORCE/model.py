import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, action_size=2):
        super().__init__()
        # 4 是状态数组的维度
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    """智能体的定义"""

    def __init__(self):
        self.gamma = 0.98  # 折扣因子
        self.lr = 0.0002
        self.action_size = 2  # 两个动作
        self.baseline = 50

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        """π(a|s)"""
        state = torch.tensor(state[np.newaxis, :])
        probs = self.pi(state)  # 求出动作的概率分布
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()

        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        # for reward, _ in reversed(self.memory):
        #     G = reward + self.gamma * G
        # for _, prob in self.memory:
        #     loss += - prob.log() * G
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += - prob.log() * (G - self.baseline)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
