import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

print(gym.__version__)
# 强化学习之父：Richard Sutton
env = gym.make("CartPole-v0")
state = env.reset()
done = False
# 将每一步的即时奖励R_t都保存在一个列表中
episode_rewards = []
G = 0
frames = []
gamma = 0.95

# # 前向过程：玩一局游戏，采样一条轨迹τ(trajectory)
while not done:
    frames.append(env.render(mode="rgb_array"))
    action = random.choice([0, 1])
    _, reward, done, _ = env.step(action)
    episode_rewards.append(reward)

for r in episode_rewards[::-1]:
    G = r + gamma * G
print("total_rewards:",G)
env.close()

def show_animation(imgs):
    rc("animation", html="jshtml")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    frames = []

    ax.text(10, 20, "", fontsize=12, color="black")

    for i, img in enumerate(imgs):
        frame = [ax.imshow(img, animated=True)]
        frame.append(ax.text(10, 20, f"Step: {i+1}", animated=True))  # Step数表示
        frames.append(frame)

    ax.axis("off")

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    # 保存动画
    ani.save("cartpole.mp4", writer="ffmpeg")
    ani.save("cartpole.gif", writer="pillow")

    plt.close(fig)
    return ani


show_animation(frames)
