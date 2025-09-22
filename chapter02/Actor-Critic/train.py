import matplotlib.pyplot as plt
import gym
from model import Agent
import torch

env = gym.make('CartPole-v0')
agent = Agent()
reward_history = []

for episode in range(2000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))
torch.save(agent.pi.state_dict(), 'policy_model.pth')
print("模型已保存")

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()