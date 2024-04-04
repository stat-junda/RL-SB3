# https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
import gym
import pygame
from stable_baselines3 import PPO #A2C

env = gym.make('LunarLander-v2', render_mode='human')  # continuous: LunarLanderContinuous-v2
env.reset()

model = PPO('MlpPolicy', env, verbose=1) #verbose=1表示在训练过程中打印一些信息
model.learn(total_timesteps=100000) #progress_bar=True表示显示训练进度条

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)
    
    pygame.quit()  # 在每个episode结束后关闭Pygame窗口
  
env.close()