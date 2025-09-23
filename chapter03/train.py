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


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(
            total=int(num_episodes/10),
            desc='Iteration %d' % i
        ) as pbar:
            # 循环 500 / 10 = 50
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                # 采样一条轨迹，记录了每个动作的相关信息
                # 注意：这条轨迹被反复使用10次，样本利用率较高
                while not done:
                    # 每执行一次动作，就将相关信息添加到对应的数组
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                # 更新策略 ---> 冻结一份旧策略的概率，然后更新10次策略网络
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (
                            num_episodes/10 * i + i_episode+1
                        ),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


# 训练策略
return_list = train_on_policy_agent(env, agent, num_episodes)
torch.save(agent.critic.state_dict(), 'critic_model.pth')
torch.save(agent.actor.state_dict(), 'actor_model.pth')


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('CartPole-v0')
plt.show()
