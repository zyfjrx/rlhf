import torch
from ppo import PPO
import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
num_episodes = 500  # 玩500个回合的游戏
hidden_dim = 128
gamma = 0.98  # 折扣因子
lmbda = 0.95  # 广义优势估计用到的超参数λ
epochs = 10  # 智能体每次更新策略需要训练10轮
eps = 0.2  # ϵ = 0.2
device = torch.device("cuda") \
    if torch.cuda.is_available() \
    else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
# state_dim = 4
state_dim = env.observation_space.shape[0]
# action_dim = 2
action_dim = env.action_space.n
# 初始化智能体
agent = PPO(
    state_dim,
    hidden_dim,
    action_dim,
    actor_lr,
    critic_lr,
    lmbda,
    epochs,
    eps,
    gamma,
    device
)

agent.critic.load_state_dict(torch.load('critic_model.pth'))
agent.actor.load_state_dict(torch.load('actor_model.pth'))

agent.actor.eval()
agent.critic.eval()

def test_render(agent, env, episodes=5):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        print(f"回合 {episode + 1}: 总奖励 = {total_reward}")
    env.close()


test_render(agent, env)
