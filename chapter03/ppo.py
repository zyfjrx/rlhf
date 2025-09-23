import torch
from model import PolicyNet, ValueNet
import torch.nn.functional as F


def compute_advantage(gamma, lmbda, td_delta):
    """ 广义优势估计 """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(
        self,
        state_dim,  # 状态的维度 = 4
        hidden_dim,  # 128
        action_dim,  # 2个动作
        actor_lr,  # 策略（演员）的学习率
        critic_lr,  # 价值网络（评论家）的学习率
        lmbda,  # λ，广义优势估计的超参数
        epochs,  # 训练轮数
        eps,  # ϵ，裁剪范围超参数(1-ϵ,1+ϵ)
        gamma,  # γ折扣因子
        device  # gpu or cpu
    ):
        # 演员是策略网络
        self.actor = PolicyNet(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)
        # 评论家是价值网络
        self.critic = ValueNet(
            state_dim,
            hidden_dim
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

        # 采取动作
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

        # 更新策略，每次智能体更新策略，都需要训练epochs轮
    def update(self, transition_dict):
        # s_t 当前的状态
        states = torch.tensor(
            transition_dict['states'],
            dtype=torch.float
        ).to(self.device)
        # a_t 采取的动作
        actions = torch.tensor(
            transition_dict['actions']
        ).view(-1, 1).to(self.device)
        # r_t：t时刻获得的即时奖励
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.float
        ).view(-1, 1).to(self.device)
        # s_{t+1} 下一个状态
        next_states = torch.tensor(
            transition_dict['next_states'],
            dtype=torch.float
        ).to(self.device)
        # 是否结束标志位
        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.float
        ).view(-1, 1).to(self.device)
        # td_target = R_t + γV(S_{t+1})
        td_target = rewards + \
            self.gamma * self.critic(next_states) * (1 - dones)
        # δ = R_t + γV(S_{t+1}) - V(S_t)
        td_delta = td_target - self.critic(states)
        # A^{π_θ_old}_t
        # `compute_advantage` 计算广义优势估计：n步TD误差
        advantage = compute_advantage(
            self.gamma,
            self.lmbda,
            td_delta.cpu()
        ).to(self.device)

        # 旧策略采取动作的 **对数概率** log{π_θ_old(a_t|s_t)}
        old_log_probs = torch.log(
            # todo: gather讲解
            self.actor(states).gather(1, actions)
        ).detach()
        # 训练epochs轮
        for _ in range(self.epochs):
            # 新策略采取动作的 **对数概率** log{π_θ_new(a_t|s_t)}
            # 第1轮训练的时候，新旧策略模型是相同模型
            # 新的策略用旧策略产生的轨迹的状态输出动作的概率
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 比率计算公式
            ratio = torch.exp(log_probs - old_log_probs)
            # p_t(θ)*优势，公式中逗号的左边项
            surr1 = ratio * advantage
            # 公式中逗号的右边项
            surr2 = torch.clamp(
                ratio,  # p_t(θ)
                1 - self.eps,  # 1-ϵ = 0.8
                1 + self.eps  # 1+ϵ = 1.2
            ) * advantage
            # PPO损失函数
            # 公式中min的实现 `-torch.min(surr1, surr2)`
            # 策略的损失
            # 每个时刻新旧策略采取动作a_t的比率的 **均值**
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失
            # 每一步的TD误差的平均值
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
