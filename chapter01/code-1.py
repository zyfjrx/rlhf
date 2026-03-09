import gym

print(gym.__version__)
# 强化学习之父：Richard Sutton
env = gym.make("CartPole-v0")
state = env.reset() # 重置为S_0状态
print("S_0: ", state)

action_space = env.action_space # 两个动作
print(action_space)

# 选择动作：向左推
action = 0
# 采取向左推的动作
next_state, reward, done, info = env.step(action)
print("S_1: ", next_state)
print("R_0: ", reward)
print("是否结束：", done)
