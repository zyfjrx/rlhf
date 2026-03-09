import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



def show_animation(imgs):
    rc("animation", html="jshtml")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    frames = []

    text = ax.text(10, 20, "", fontsize=12, color="black")

    for i, img in enumerate(imgs):
        frame = [ax.imshow(img, animated=True)]
        frame.append(ax.text(10, 20, f"Step: {i + 1}", animated=True))  # Step数表示
        frames.append(frame)

    ax.axis("off")

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    # 保存动画
    ani.save("cartpole.mp4", writer="ffmpeg")
    ani.save("cartpole.gif", writer="pillow")

    plt.close(fig)
    return ani


def plot_loss(episode_list, return_list, filename):
    """绘制奖励图像"""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


class PolicyNet(nn.Module):
    """策略神经网络的结构"""

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):  # x是S_t
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x

class ValueNet(nn.Module):
    """价值神经网络的结构"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):  # x是S_t
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98  # 折扣因子γ
        self.lr_pi = 0.0002  # 学习率α
        self.lr_v = 0.005  # 学习率α
        self.action_size = 2  # 两个动作：向左推和向右推
        # 初始化策略网络π_θ
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(
            self.pi.parameters(),
            lr=self.lr_pi
        )
        self.optimizer_v = optim.Adam(
            self.v.parameters(),
            lr=self.lr_v
        )

    def get_action(self, state):
        """输入参数：环境的状态"""
        # 动作的概率分布
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        # 创建一个二项分布采样器
        m = Categorical(probs)
        # 采样一个动作
        action = m.sample().item()

        return action, probs


    def update(self, state,next_state, reward, action_prob, done):
        state = torch.tensor(state).unsqueeze(0)
        next_state = torch.tensor(next_state).unsqueeze(0)

        # 计算价值网络的损失
        # TD目标
        # td误差δ = R_t + γV(S_(t+1)) - V(S_t)
        # TD target = δ + V(S_t)
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        # 从计算图中剥离，不参与反向传播（target作为一个常数）
        target.detach()
        v = self.v(state)
        loss_fn = nn.MSELoss()
        # mse要求目标为常数
        loss_v = loss_fn(v, target)

        # 策略网络的损失
        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.detach().item()

        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_pi.step()
        self.optimizer_v.step()


env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(3000):
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, probs = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        # 执行完一个动作，立即更新策略网络和价值网络
        # 相比蒙特卡洛不需要采样完整个轨迹来更新
        agent.update(state, next_state, reward, probs[action], done)
        state = next_state
        total_reward += reward
    return_list.append(total_reward)
    episode_list.append(episode)
    if episode % 100 == 0:
        print(f"回合：{episode}, 总奖励：{total_reward}")

# 可视化损失
plot_loss(episode_list, return_list, "actor-critic-pg-loss.pdf")



